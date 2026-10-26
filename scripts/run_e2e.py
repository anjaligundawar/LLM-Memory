"""End-to-end QA accuracy under pollution, with and without hygiene.

Answers a sample of questions with an LLM reader over the top-k retrieved
memories and grades them with an LLM judge. Costs API credits: start with
a small --n-queries and check the printed estimate.

    python scripts/run_e2e.py configs/e2e.yaml
"""
import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memhygiene.data import load_corpus  # noqa: E402
from memhygiene.data.loghub import routine_lines  # noqa: E402
from memhygiene.embed import encode_docs, encode_queries, make_embedder  # noqa: E402
from memhygiene.features import store_features  # noqa: E402
from memhygiene.hygiene import HygieneConfig, gate_from_config, hard_keep_mask, run_hygiene, soft_rerank  # noqa: E402
from memhygiene.llm import Claude  # noqa: E402
from memhygiene.pollution import Polluter  # noqa: E402
from memhygiene.reader import answer, judge  # noqa: E402
from memhygiene.retrieval import make_retriever  # noqa: E402
from memhygiene.runner import _make_nli  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cache = Path(cfg.get("cache_dir", "cache"))
    emb = make_embedder(cfg.get("embedder", {}), cache_dir=cache)
    hcfg = HygieneConfig(**cfg.get("hygiene", {}))
    nli = _make_nli(cfg, cache)
    llm = Claude(**cfg.get("llm", {}), cache_dir=cache)
    k, rname = cfg.get("k", 10), cfg.get("retriever", "hybrid")
    rows = []
    for cname, ckw in cfg["corpora"].items():
        corpus = load_corpus(cname, **(ckw or {}))
        gate = gate_from_config(cfg.get("gate"), cname)
        pol = Polluter(corpus, transient_pool=routine_lines(corpus) if cname == "loghub" else None)
        rng = random.Random(0)
        pairs = [(s, q) for s in corpus.stores for q in s.queries if q.answer]
        pairs = rng.sample(pairs, min(cfg.get("n_queries", 100), len(pairs)))
        by_store = {}
        for s, q in pairs:
            by_store.setdefault(s.id, (s, []))[1].append(q)
        n_calls = len(pairs) * len(cfg["conditions"]) * len(cfg["modes"]) * 2
        print(f"{cname}: {len(pairs)} questions, about {n_calls} LLM calls (cached calls are free)")
        for jtype, level in cfg["conditions"]:
            for s, qs in by_store.values():
                ps = pol.pollute(s, jtype, level, seed=0)
                units = ps.units
                V = encode_docs(emb, [u.text for u in units])
                full = run_hygiene(units, V, hcfg, nli)
                gated, _ = gate.apply(full, store_features(units, V, hcfg.tau_dup, hcfg.tau_sup))
                keep = hard_keep_mask(full.scores)
                umap = ps.unit_map()
                r_all = make_retriever(rname).index(units, V)
                r_hard = make_retriever(rname).index([u for u, m in zip(units, keep) if m], V[keep])
                qv = encode_queries(emb, [q.text for q in qs])
                for q, vec in zip(qs, qv):
                    cand = r_all.search(q.text, vec, k=50)
                    ranked = {"none": [u for u, _ in cand[:k]],
                              "hard": [u for u, _ in r_hard.search(q.text, vec, k=k)],
                              "soft": [u for u, _ in soft_rerank(cand, {u.id: float(x) for u, x in zip(units, full.scores)}, k=k)],
                              "gated": [u for u, _ in soft_rerank(cand, {u.id: float(x) for u, x in zip(units, gated.scores)}, k=k)]}
                    for mode in cfg["modes"]:
                        mems = [umap[i] for i in ranked[mode]]
                        pred = answer(llm, q.text, mems, q.timestamp)
                        ok = judge(llm, q.text, q.answer, pred)
                        rows.append(dict(corpus=cname, store=s.id, query=q.id, qtype=q.qtype, jtype=jtype,
                                         level=level, retriever=rname, hygiene=mode, pred=pred, gold=q.answer,
                                         correct=ok, junk_in_context=sum(m.is_junk for m in mems)))
            print(f"  {cname} {jtype} {level} done")
    out = Path(cfg.get("out", "results/e2e"))
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(out / "e2e.parquet", index=False)
    print(df.groupby(["corpus", "jtype", "level", "hygiene"]).correct.mean().unstack("hygiene").round(3))
