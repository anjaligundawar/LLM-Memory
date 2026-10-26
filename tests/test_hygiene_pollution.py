"""Focused tests for Module 3: junk-memory injection (pollution/) and the
hygiene layer (hygiene/). They pin down the contracts the experiment grid
relies on, so a change to either package cannot silently break the results:

  * pollution never touches genuine memories and labels every junk unit
  * each junk type has the timestamp / parent relationship its definition needs
  * each hygiene component flags what it should and keeps the right copy
  * the gate switches off over-eager components and enforces its budget
"""
import random
from collections import Counter

import numpy as np
import pytest

from memhygiene.embed import HashingEmbedder
from memhygiene.features import FEATURES, store_features
from memhygiene.hygiene import Gate, GateModel, HygieneConfig, combine, hard_keep_mask, run_hygiene
from memhygiene.hygiene.nli import OverlapContradiction
from memhygiene.pollution import JUNK_TYPES, Polluter
from memhygiene.pollution.synthetic import TRANSIENT_FALLBACK, _mutate
from memhygiene.schema import Corpus, MemoryUnit, Store
from memhygiene.text import informativeness

DERIVED = {"duplicate", "stale", "contradictory"}       # junk that copies or edits a genuine unit
FREE = {"irrelevant", "temporary"}                       # junk brought in from elsewhere


def _units(sid, texts, t0=0.0):
    return [MemoryUnit(id=f"{sid}/u{i}", text=t, timestamp=t0 + 100.0 * i, store_id=sid)
            for i, t in enumerate(texts)]


@pytest.fixture
def two_people():
    """Two stores with disjoint texts, so junk provenance can be checked."""
    a = Store("p/a", _units("p/a", [
        "Priya moved to Bangalore in March 2023 to join Infosys.",
        "Priya adopted a cat named Momo in November 2023.",
        "Priya switched jobs and joined Flipkart in January 2024.",
        "Priya is learning guitar from a teacher named Arjun.",
    ]), [])
    b = Store("p/b", _units("p/b", [
        "Rohan lives in Pune and works at VoltRide.",
        "Rohan bought a blue Ather scooter in May 2023.",
        "Rohan did the Kedarkantha trek with 12 friends.",
        "haha yes",
    ]), [])
    return Corpus("people", [a, b])


# ---------------------------------------------------------------- pollution

@pytest.mark.parametrize("jtype,level", [("none", 0.5), ("mixed", 0.0), ("stale", -1.0)])
def test_no_pollution_returns_clean_copy(corpus, jtype, level):
    s = corpus.stores[0]
    out = Polluter(corpus, seed=0).pollute(s, jtype, level)
    assert [u.id for u in out.units] == [u.id for u in s.units]
    assert out.units is not s.units                     # a copy, not the same list


def test_pollute_does_not_mutate_input_store(corpus):
    s = corpus.stores[0]
    before = [(u.id, u.text, u.timestamp, u.junk_type) for u in s.units]
    Polluter(corpus, seed=0).pollute(s, "mixed", 0.75, seed=2)
    assert [(u.id, u.text, u.timestamp, u.junk_type) for u in s.units] == before


@pytest.mark.parametrize("jtype", JUNK_TYPES)
def test_junk_units_are_labelled_and_uniquely_named(corpus, jtype):
    s = corpus.stores[0]
    out = Polluter(corpus, seed=0).pollute(s, jtype, 0.5, seed=4)
    junk = [u for u in out.units if u.is_junk]
    assert junk and all(u.junk_type == jtype for u in junk)
    assert all(u.store_id == s.id and u.id.startswith(f"{s.id}/junk") for u in junk)
    ids = [u.id for u in out.units]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("jtype", JUNK_TYPES)
def test_junk_parent_links(corpus, jtype):
    s = corpus.stores[0]
    genuine = {u.id for u in s.units}
    junk = [u for u in Polluter(corpus, seed=0).pollute(s, jtype, 0.5, seed=5).units if u.is_junk]
    if jtype in DERIVED:
        assert all(u.parent_id in genuine for u in junk)
    else:
        assert all(u.parent_id is None for u in junk)


def test_duplicates_come_after_their_original(corpus):
    s = corpus.stores[0]
    m = s.unit_map()
    junk = [u for u in Polluter(corpus, seed=0).pollute(s, "duplicate", 0.5, seed=6).units if u.is_junk]
    assert all(u.timestamp > m[u.parent_id].timestamp for u in junk)


@pytest.mark.parametrize("jtype", sorted(FREE))
def test_free_junk_lands_inside_the_store_timeline(two_people, jtype):
    s = two_people.stores[0]
    lo, hi = min(u.timestamp for u in s.units), max(u.timestamp for u in s.units)
    junk = [u for u in Polluter(two_people, seed=0).pollute(s, jtype, 1.0, seed=1).units if u.is_junk]
    assert all(lo <= u.timestamp <= hi for u in junk)


def test_irrelevant_junk_is_borrowed_from_another_store(two_people):
    a, b = two_people.stores
    junk = [u for u in Polluter(two_people, seed=0).pollute(a, "irrelevant", 1.0, seed=2).units if u.is_junk]
    other = {u.text for u in b.units}
    assert junk and all(u.text in other for u in junk)


def test_temporary_junk_is_low_information(two_people):
    a = two_people.stores[0]
    junk = [u for u in Polluter(two_people, seed=0).pollute(a, "temporary", 1.0, seed=3).units if u.is_junk]
    assert junk
    for u in junk:
        assert informativeness(u.text) <= 0.25 or u.text in TRANSIENT_FALLBACK


def test_mixed_spreads_junk_evenly_over_all_types(corpus):
    s = corpus.stores[0]                                # 10 genuine units -> 10 junk at L=1.0
    junk = [u for u in Polluter(corpus, seed=0).pollute(s, "mixed", 1.0, seed=0).units if u.is_junk]
    assert Counter(u.junk_type for u in junk) == {t: 2 for t in JUNK_TYPES}


def test_different_seeds_give_different_junk(corpus):
    s, p = corpus.stores[0], Polluter(corpus, seed=0)
    runs = {tuple(u.text for u in p.pollute(s, "mixed", 0.5, seed=k).units) for k in range(5)}
    assert len(runs) > 1


def test_mutate_changes_one_value_and_keeps_speaker_prefix():
    rng = random.Random(0)
    assert _mutate("lol ok", rng, ["Infosys"], negate_ok=False) is None   # nothing to change
    src = "Priya: I joined Infosys in 2023."
    for _ in range(20):
        out = _mutate(src, rng, ["Infosys", "Flipkart"], negate_ok=True)
        assert out is not None and out != src and out.startswith("Priya: ")


# ------------------------------------------------------------------ hygiene

def _pair(text_old, text_new, cos):
    """Two units (older first) with chosen cosine similarity between them."""
    units = [MemoryUnit("s/u0", text_old, 1.0, "s"), MemoryUnit("s/u1", text_new, 2.0, "s")]
    V = np.array([[1.0, 0.0], [cos, np.sqrt(1 - cos ** 2)]], dtype=np.float32)
    return units, V


def test_dedup_flags_the_later_copy_only():
    units, V = _pair("Priya joined Flipkart.", "Priya joined Flipkart.", 1.0)
    res = run_hygiene(units, V, HygieneConfig(), components=["dedup"])
    assert res.by_component["dedup"].tolist() == [0.0, 1.0]
    assert res.reasons == [None, "dedup"]


def test_dedup_tie_on_timestamp_keeps_first_position():
    units = [MemoryUnit(f"s/u{i}", "same text", 5.0, "s") for i in range(2)]
    res = run_hygiene(units, np.ones((2, 2), dtype=np.float32) / np.sqrt(2), HygieneConfig(),
                      components=["dedup"])
    assert res.by_component["dedup"].tolist() == [0.0, 1.0]


def test_supersession_invalidates_the_older_version():
    units, V = _pair("Priya works at Infosys since 2023.", "Priya works at Flipkart since 2024.", 0.85)
    res = run_hygiene(units, V, HygieneConfig(), components=["supersession"])
    s = res.by_component["supersession"]
    assert s[0] == pytest.approx(0.85, abs=1e-5) and s[1] == 0.0


def test_supersession_ignores_pairs_with_identical_values():
    units, V = _pair("Priya works at Infosys.", "Priya really works at Infosys.", 0.85)
    res = run_hygiene(units, V, HygieneConfig(), components=["supersession"])
    assert res.by_component["supersession"].sum() == 0


def test_contradiction_needs_an_nli_model_and_flags_the_older_claim():
    units, V = _pair("Priya is moving to Pune.", "Priya is not moving to Pune.", 0.8)
    assert run_hygiene(units, V, HygieneConfig(), components=["contradiction"]).scores.sum() == 0
    res = run_hygiene(units, V, HygieneConfig(), nli=OverlapContradiction(), components=["contradiction"])
    c = res.by_component["contradiction"]
    assert c[0] >= 0.5 and c[1] == 0.0


def test_transient_flags_small_talk_not_facts(corpus):
    s = corpus.stores[0]
    V = HashingEmbedder().encode([u.text for u in s.units])
    res = run_hygiene(s.units, V, HygieneConfig(), components=["transient"])
    flagged = {u.text for u, f in zip(s.units, res.flagged()) if f}
    assert {"lol ok", "hey, how are you doing today?"} <= flagged
    assert "Priya adopted a cat named Momo in November 2023." not in flagged


def test_disabled_components_are_not_run(corpus):
    s = corpus.stores[0]
    V = HashingEmbedder().encode([u.text for u in s.units])
    res = run_hygiene(s.units, V, HygieneConfig(dedup=False, transient=False))
    assert "dedup" not in res.by_component and "transient" not in res.by_component
    assert "isolation" not in res.by_component                   # off by default
    assert len(res.scores) == len(s.units)


def test_empty_store_is_handled():
    res = run_hygiene([], np.zeros((0, 4), dtype=np.float32), HygieneConfig())
    assert len(res.scores) == 0 and res.reasons == []


def test_combine_takes_the_max_and_records_its_reason():
    res = combine({"dedup": np.array([0.2, 0.9, 0.0]), "transient": np.array([0.7, 0.4, 0.0])},
                  ["dedup", "transient"])
    assert res.scores.tolist() == pytest.approx([0.7, 0.9, 0.0])
    assert res.reasons == ["transient", "dedup", None]


def test_hard_keep_mask_threshold():
    assert hard_keep_mask(np.array([0.0, 0.49, 0.5, 1.0])).tolist() == [True, True, False, False]


# --------------------------------------------------------------------- gate

def test_gate_keeps_a_well_behaved_component():
    full = combine({"transient": np.array([1.0] + [0.0] * 9)}, ["transient"])
    gated, on = Gate().apply(full, {})
    assert on == {"transient": True}
    assert gated.scores.tolist() == full.scores.tolist()


def test_gate_budget_keeps_only_the_strongest_flags():
    full = combine({"dedup": np.array([0.6, 0.9, 0.7, 0.8])}, ["dedup"])
    gated, on = Gate(max_flag_rate=1.0, budget=0.5).apply(full, {})
    assert on["dedup"] is True
    assert gated.by_component["dedup"].tolist() == pytest.approx([0.0, 0.9, 0.0, 0.8])


def test_gate_model_round_trip_and_clipped_prediction(tmp_path):
    k = len(FEATURES) + 1
    m = GateModel(coef={"dedup": [5.0] * k}, intercept={"dedup": 0.0}, mean=[0.0] * k, std=[1.0] * k)
    m.save(tmp_path / "gate.json")
    m2 = GateModel.load(tmp_path / "gate.json")
    assert m2.coef == m.coef
    pred = m2.predict({f: 1.0 for f in FEATURES}, {"dedup": 1.0})
    assert pred == {"dedup": 1.0}                                  # clipped to [0, 1]
    _, on = Gate(model=m2).apply(combine({"dedup": np.zeros(4)}, ["dedup"]), {f: 1.0 for f in FEATURES})
    assert on["dedup"] is False


# ----------------------------------------------------------------- features

def test_store_features_shape_and_small_store(corpus):
    s = corpus.stores[0]
    V = HashingEmbedder().encode([u.text for u in s.units])
    f = store_features(s.units, V)
    assert list(f) == FEATURES and all(np.isfinite(v) for v in f.values())
    assert 0.0 <= f["dup_pair_rate"] <= f["template_rate"] <= 1.0
    assert store_features(s.units[:1], V[:1]) == {k: 0.0 for k in FEATURES}
