import numpy as np
import pandas as pd

from memhygiene.data import longmemeval
from memhygiene.embed import HashingEmbedder
from memhygiene.eval.bootstrap import cluster_bootstrap_diff, cluster_bootstrap_mean
from memhygiene.eval.metrics import audc, query_metrics
from memhygiene.features import store_features
from memhygiene.hygiene import Gate, HygieneConfig, run_hygiene, soft_rerank
from memhygiene.hygiene.nli import OverlapContradiction
from memhygiene.pollution import JUNK_TYPES, Polluter
from memhygiene.retrieval import make_retriever
from memhygiene.text import informativeness, slots


def test_longmemeval_loader(lme_file):
    c = longmemeval.load(lme_file)
    assert len(c.stores) == 1                      # abstention question dropped
    s = c.stores[0]
    assert len(s.units) == 4 and len(s.queries[0].gold_ids) == 2


def test_text_helpers():
    assert {"2023", "march", "infosys", "bangalore"} <= slots("Priya moved to Bangalore in March 2023 to join Infosys")
    assert informativeness("lol ok") < 0.2 < informativeness("Priya adopted a cat named Momo in November")


def test_pollution_counts_and_labels(corpus):
    p = Polluter(corpus, targeted=1.0, seed=0)
    s = corpus.stores[0]
    for jt in JUNK_TYPES + ["mixed"]:
        ps = p.pollute(s, jt, 0.5, seed=1)
        junk = [u for u in ps.units if u.is_junk]
        assert len(junk) == 5, jt
        assert [u.id for u in ps.units if not u.is_junk] == [u.id for u in s.units]
    stale = [u for u in p.pollute(s, "stale", 0.5).units if u.is_junk]
    m = s.unit_map()
    for u in stale:
        assert u.timestamp < m[u.parent_id].timestamp and u.text != m[u.parent_id].text
    contra = [u for u in p.pollute(s, "contradictory", 0.5).units if u.is_junk]
    assert all(u.timestamp > m[u.parent_id].timestamp for u in contra)
    # deterministic
    assert [u.text for u in p.pollute(s, "mixed", 0.5, 3).units] == [u.text for u in p.pollute(s, "mixed", 0.5, 3).units]


def test_retrievers_find_gold(corpus):
    emb = HashingEmbedder()
    s = corpus.stores[0]
    V = emb.encode([u.text for u in s.units])
    for name in ["dense", "bm25", "hybrid", "graph"]:
        r = make_retriever(name, use_spacy=False) if name == "graph" else make_retriever(name)
        r.index(s.units, V)
        q = s.queries[1]
        top = [u for u, _ in r.search(q.text, emb.encode([q.text])[0], k=3)]
        assert q.gold_ids[0] in top, name


def test_hygiene_flags_exact_duplicate_and_keeps_original(corpus):
    s = corpus.stores[0]
    p = Polluter(corpus, seed=0)
    ps = p.pollute(s, "duplicate", 0.3, seed=0)
    emb = HashingEmbedder()
    V = emb.encode([u.text for u in ps.units])
    res = run_hygiene(ps.units, V, HygieneConfig(), nli=OverlapContradiction())
    by_id = dict(zip([u.id for u in ps.units], res.scores))
    reason = dict(zip([u.id for u in ps.units], res.reasons))
    exact = [u for u in ps.units if u.is_junk and u.text == s.unit_map()[u.parent_id].text]
    assert exact
    for u in exact:
        assert by_id[u.id] >= 0.5 and reason[u.parent_id] != "dedup"
    assert res.scores[[i for i, u in enumerate(ps.units) if u.text == "lol ok"][0]] >= 0.5   # transient


def test_gate_disables_overeager_component(corpus):
    s = corpus.stores[0]
    emb = HashingEmbedder()
    V = emb.encode([u.text for u in s.units])
    res = run_hygiene(s.units, V, HygieneConfig(contradiction=False))
    res.by_component["dedup"][:] = 1.0                       # pretend dedup flags everything
    gated, on = Gate(max_flag_rate=0.5).apply(res, store_features(s.units, V))
    assert on["dedup"] is False and gated.by_component["dedup"].sum() == 0


def test_soft_rerank_demotes_invalidated():
    ranked = [("a", 3.0), ("b", 2.0), ("c", 1.0)]
    out = soft_rerank(ranked, {"a": 1.0}, lam=0.9, k=3)
    assert [u for u, _ in out][0] == "b"


def test_metrics_and_bootstrap():
    m = query_metrics(["x", "g1", "j", "g2"], ["g1", "g2"], {"j"}, ks=(1, 3))
    assert m["mrr"] == 0.5 and m["recall@3"] == 0.5 and abs(m["junk@3"] - 1 / 3) < 1e-9
    assert abs(audc([0, 0.5, 1.0], [1, 1, 0]) - 0.75) < 1e-9
    rng = np.random.default_rng(0)
    a = rng.normal(1, 1, 200)
    r = cluster_bootstrap_diff(a, np.zeros(200), clusters=np.repeat(np.arange(40), 5))
    assert r["lo"] > 0 and r["p"] < 0.05
    r = cluster_bootstrap_mean(np.ones(10))
    assert r["lo"] == r["hi"] == 1.0


def test_runner_on_toy(tmp_path, corpus, monkeypatch):
    import memhygiene.runner as runner
    monkeypatch.setattr(runner, "load_corpus", lambda name, **kw: corpus)
    cfg = dict(out=str(tmp_path / "res"), cache_dir=str(tmp_path / "cache"), embedder={"name": "hashing"},
               nli={"model": "overlap"}, corpora={"toy": {}},
               pollution={"types": ["duplicate", "mixed"], "levels": [0.0, 0.5], "seeds": [0]},
               retrievers=["dense", "bm25"])
    out = runner.run(cfg)
    q = runner.load_results(out)
    h = runner.load_results(out, "hygiene")
    assert set(q.hygiene) == {"none", "hard", "soft", "gated"}
    assert len(q) == 3 * 2 * 2 * 4 * 2          # cells x stores x queries x modes x retrievers
    assert (h[h.component == "ALL"].n_junk > 0).any()
    from memhygiene import analysis as A
    assert not A.hygiene_effect(q).empty and not A.hygiene_rates(h).empty


def test_mab_loader(tmp_path):
    from memhygiene.data import mab_conflict
    ctx = "Here is a list of facts:\n0. The capital of Zorland is Arn.\n1. Tom works for Acme.\n2. The capital of Zorland is Bel.\n"
    df = pd.DataFrame([{"context": ctx, "questions": ["What is the capital of Zorland?", "Who owns Mars?"],
                        "answers": [["Bel"], ["nobody"]], "metadata": None}])
    p = tmp_path / "cr.parquet"
    df.to_parquet(p)
    c = mab_conflict.load(p)
    s = c.stores[0]
    assert len(s.units) == 3 and len(s.queries) == 1 and s.queries[0].gold_ids == ["mab/0/f2"]
