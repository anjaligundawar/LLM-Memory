"""Regression tests for bugs found in a code review. Each test failed before
its fix."""

import numpy as np
import pandas as pd

from memhygiene import analysis as A
from memhygiene.hygiene import Gate, GateModel, gate_from_config
from memhygiene.features import FEATURES
from memhygiene.pollution import natural
from memhygiene.retrieval import graph
from memhygiene.schema import MemoryUnit


class _FakeLLM:
    def __init__(self):
        self.prompts = []

    def structured(self, system, user, schema):
        self.prompts.append(user)
        return schema(label="valid")


def test_annotate_never_lists_target_as_its_own_neighbour():
    texts = ["Alice likes tea.", "Bob owns a red bike.", "Carol moved to Pune."]
    units = [MemoryUnit(f"s/{i}", t, 1.7e9 + i * 86400, "s") for i, t in enumerate(texts)]
    llm = _FakeLLM()
    natural.annotate(llm, units, np.eye(3, dtype=np.float32), neighbours=6)   # store smaller than k
    for t, prompt in zip(texts, llm.prompts):
        others = prompt.split("Other memories:")[1]
        assert t not in others
        assert sum(o in others for o in texts if o != t) == 2


def test_junk_reduction_is_nan_when_there_was_no_junk():
    rows = [dict(corpus="c", store="s", query="q", retriever="dense", jtype="none", level=0.0, hygiene=m,
                 **{"recall@5": 1.0, "junk@5": 0.0}) for m in ("none", "hard")]
    e = A.hygiene_effect(pd.DataFrame(rows), modes=("hard",))
    assert np.isnan(e.junk_reduction.iloc[0])
    rows = [dict(r, jtype="mixed", level=0.5, **{"junk@5": 0.4 if r["hygiene"] == "none" else 0.0})
            for r in rows]
    assert np.isinf(A.hygiene_effect(pd.DataFrame(rows), modes=("hard",)).junk_reduction.iloc[0])


def test_vector_vs_graph_without_graph_runs_is_empty_not_a_crash():
    rows = [dict(corpus="c", store="s", query="q", retriever=r, jtype=j, level=lvl, seed=0, hygiene="none",
                 **{"recall@5": v})
            for r in ("dense", "bm25") for j, lvl, v in (("none", 0.0, 1.0), ("mixed", 0.5, 0.5))]
    assert A.vector_vs_graph(pd.DataFrame(rows)).empty


def test_gate_from_config_accepts_main_yaml_keys(tmp_path):
    k = len(FEATURES) + 1
    m = GateModel(coef={"dedup": [0.0] * k}, intercept={"dedup": 0.0}, mean=[0.0] * k, std=[1.0] * k)
    m.save(tmp_path / "loco_locomo.json")
    m.save(tmp_path / "all.json")
    g = gate_from_config({"epsilon": 0.1, "budget": 0.3, "max_flag_rate": 0.4, "model_dir": str(tmp_path)},
                         "locomo")
    assert isinstance(g, Gate) and isinstance(g.model, GateModel)
    assert (g.epsilon, g.budget, g.max_flag_rate) == (0.1, 0.3, 0.4)
    assert gate_from_config({"model": str(tmp_path / "all.json")}).model.coef == m.coef
    assert gate_from_config(None).model is None
    assert gate_from_config({"model_dir": str(tmp_path)}).model is None   # no corpus -> rule gate


def test_key_cache_keeps_the_batch_it_just_computed(monkeypatch):
    calls = []
    real = graph.entities
    monkeypatch.setattr(graph, "entities", lambda t: calls.append(t) or real(t))
    monkeypatch.setattr(graph, "_KEY_CACHE", {})
    monkeypatch.setattr(graph, "_KEY_CACHE_MAX", 3)
    graph.extract_keys(["a b c", "d e f"], use_spacy=False)
    out = graph.extract_keys(["Priya joined Flipkart.", "Rohan lives in Pune."], use_spacy=False)
    assert calls[-2:] == ["Priya joined Flipkart.", "Rohan lives in Pune."]   # each computed once
    assert len(calls) == 4
    assert "e:priya" in out[0] and "e:pune" in out[1]
    assert set(graph._KEY_CACHE) == {("regex", "Priya joined Flipkart."), ("regex", "Rohan lives in Pune.")}
