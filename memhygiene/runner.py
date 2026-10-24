"""Experiment grid: corpus x junk type x level x seed x retriever x hygiene mode.

Outputs (under cfg.out):
  queries/<corpus>/<type>_<level>_s<seed>.parquet   one row per query x retriever x hygiene mode
  hygiene/<corpus>/<type>_<level>_s<seed>.parquet   one row per store x hygiene mode x component
  config.yaml                                       the exact config used

A finished cell is skipped on re-run, so an interrupted grid resumes.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from .data import load_corpus
from .data.loghub import routine_lines
from .embed import encode_docs, encode_queries, make_embedder, set_threads
from .eval.metrics import query_metrics
from .features import store_features
from .hygiene import (Gate, GateModel, HygieneConfig, HygieneResult, combine, hard_keep_mask,
                      run_hygiene, soft_rerank)
from .pollution import Polluter
from .retrieval import make_retriever

MODES = ["none", "hard", "soft", "gated"]


def _make_nli(cfg: dict, cache: Path):
    name = cfg.get("nli", {}).get("model", "cross-encoder/nli-MiniLM2-L6-H768")
    if name in (None, "none"):
        return None
    if name == "overlap":
        from .hygiene.nli import OverlapContradiction
        return OverlapContradiction()
    from .hygiene.nli import NLIScorer
    return NLIScorer(name, cache_dir=cache, device=cfg.get("nli", {}).get("device"),
                     batch_size=cfg.get("nli", {}).get("batch_size", 32))


def _hygiene_rows(base: dict, units, res: HygieneResult, mode: str, gate_on=None) -> List[dict]:
    junk = np.array([u.is_junk for u in units])
    jtypes = np.array([u.junk_type or "" for u in units])
    rows = []
    comps = list(res.by_component) + ["ALL"]
    for c in comps:
        s = res.scores if c == "ALL" else res.by_component[c]
        f = s >= 0.5
        r = dict(base, hygiene=mode, component=c, n_genuine=int((~junk).sum()), n_junk=int(junk.sum()),
                 flagged_genuine=int((f & ~junk).sum()), flagged_junk=int((f & junk).sum()),
                 enabled=bool(gate_on[c]) if (gate_on is not None and c != "ALL") else True,
                 flag_rate=float(f.mean()) if len(f) else 0.0)
        for jt in sorted(set(jtypes) - {""}):
            m = jtypes == jt
            r[f"flagged_{jt}"] = int((f & m).sum())
            r[f"n_{jt}"] = int(m.sum())
        rows.append(r)
    return rows


def run(cfg: dict) -> Path:
    out = Path(cfg["out"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cache = Path(cfg.get("cache_dir", "cache"))
    print(f"using {set_threads(cfg.get('threads'))} CPU threads", flush=True)
    emb = make_embedder(cfg.get("embedder", {}), cache_dir=cache)
    hcfg = HygieneConfig(**cfg.get("hygiene", {}))
    nli = _make_nli(cfg, cache) if hcfg.contradiction else None
    gcfg = cfg.get("gate", {})
    gate = Gate(epsilon=gcfg.get("epsilon", 0.05), budget=gcfg.get("budget", 0.5),
                max_flag_rate=gcfg.get("max_flag_rate", 0.5),
                model=GateModel.load(gcfg["model"]) if gcfg.get("model") else None)
    lam = cfg.get("soft_lambda", 0.9)
    pool = cfg.get("candidate_pool", 50)
    ks = cfg.get("ks", [1, 3, 5, 10])
    pol = cfg.get("pollution", {})
    types = pol.get("types", ["irrelevant", "duplicate", "stale", "contradictory", "temporary", "mixed"])
    levels = pol.get("levels", [0.0, 0.1, 0.25, 0.5, 0.75])
    seeds = pol.get("seeds", [0, 1, 2])
    retrievers = cfg.get("retrievers", ["dense", "bm25", "hybrid", "graph"])
    modes = cfg.get("modes", MODES)

    for cname, ckw in cfg["corpora"].items():
        if gcfg.get("model_dir"):   # leave-one-corpus-out gate models from scripts/fit_gate.py
            gate.model = GateModel.load(Path(gcfg["model_dir"]) / f"loco_{cname}.json")
        corpus = load_corpus(cname, **(ckw or {}))
        print(corpus.summary(), flush=True)
        polluter = Polluter(corpus, targeted=pol.get("targeted", 0.5),
                            transient_pool=routine_lines(corpus) if cname == "loghub" else None)
        all_q = [q for s in corpus.stores for q in s.queries]
        qv = dict(zip([q.id for q in all_q], encode_queries(emb, [q.text for q in all_q])))
        (out / "queries" / cname).mkdir(parents=True, exist_ok=True)
        (out / "hygiene" / cname).mkdir(parents=True, exist_ok=True)
        for seed in seeds:
            for level in levels:
                for jtype in (["none"] if level == 0 else types):
                    tag = f"{jtype}_{level}_s{seed}"
                    qpath = out / "queries" / cname / f"{tag}.parquet"
                    if qpath.exists():
                        continue
                    t0 = time.time()
                    print(f"  {cname} {tag}: started", flush=True)
                    qrows, hrows = [], []
                    for si, store in enumerate(corpus.stores, 1):
                        ts = time.time()
                        ps = polluter.pollute(store, jtype, level, seed=seed)
                        units = ps.units
                        V = encode_docs(emb, [u.text for u in units])
                        full = run_hygiene(units, V, hcfg, nli)
                        feats = store_features(units, V, hcfg.tau_dup, hcfg.tau_sup)
                        gated, gate_on = gate.apply(full, feats)
                        base = dict(corpus=cname, store=store.id, jtype=jtype, level=level, seed=seed, **feats)
                        hrows += _hygiene_rows(base, units, full, "full")
                        hrows += _hygiene_rows(base, units, gated, "gated", gate_on)
                        junk_ids = {u.id for u in units if u.is_junk}
                        keep = hard_keep_mask(full.scores)
                        kept_units = [u for u, k in zip(units, keep) if k]
                        inv_full = {u.id: float(s) for u, s in zip(units, full.scores) if s > 0}
                        inv_gated = {u.id: float(s) for u, s in zip(units, gated.scores) if s > 0}
                        removed = {u.id for u, k in zip(units, keep) if not k}
                        for rname in retrievers:
                            r_all = make_retriever(rname).index(units, V)
                            r_hard = make_retriever(rname).index(kept_units, V[keep]) if "hard" in modes else None
                            for q in store.queries:
                                vec = qv[q.id]
                                cand = r_all.search(q.text, vec, k=max(pool, max(ks)))
                                ranked = {
                                    "none": [u for u, _ in cand[:max(ks)]],
                                    "soft": [u for u, _ in soft_rerank(cand, inv_full, lam, max(ks))],
                                    "gated": [u for u, _ in soft_rerank(cand, inv_gated, lam, max(ks))],
                                }
                                if r_hard is not None:
                                    ranked["hard"] = [u for u, _ in r_hard.search(q.text, vec, k=max(ks))]
                                gold_lost = sum(g in removed for g in q.gold_ids) / len(q.gold_ids)
                                for mode in modes:
                                    m = query_metrics(ranked[mode], q.gold_ids, junk_ids, ks)
                                    qrows.append(dict(corpus=cname, store=store.id, query=q.id, qtype=q.qtype,
                                                      jtype=jtype, level=level, seed=seed, retriever=rname,
                                                      hygiene=mode, n_gold=len(q.gold_ids),
                                                      gold_removed=gold_lost if mode == "hard" else 0.0,
                                                      **m))
                        print(f"    {cname} {tag}: store {si}/{len(corpus.stores)} ({len(units)} memories) "
                              f"{time.time() - ts:.0f}s", flush=True)
                    pd.DataFrame(hrows).to_parquet(out / "hygiene" / cname / f"{tag}.parquet", index=False)
                    pd.DataFrame(qrows).to_parquet(qpath, index=False)
                    print(f"  {cname} {tag}: {len(qrows)} rows in {time.time() - t0:.1f}s", flush=True)
    return out


def load_results(out: Path, kind: str = "queries") -> pd.DataFrame:
    files = sorted((Path(out) / kind).glob("*/*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
