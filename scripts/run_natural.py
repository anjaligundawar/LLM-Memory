"""Natural junk experiment: let an LLM memory writer build the memory store
from raw conversations, label its junk with an LLM annotator, then measure
how well hygiene separates junk from genuine memories.

    python scripts/run_natural.py configs/natural.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memhygiene.data import load_corpus  # noqa: E402
from memhygiene.embed import encode_docs, make_embedder  # noqa: E402
from memhygiene.features import store_features  # noqa: E402
from memhygiene.hygiene import HygieneConfig, gate_from_config, run_hygiene  # noqa: E402
from memhygiene.llm import Claude  # noqa: E402
from memhygiene.pollution.natural import annotate, write_memories  # noqa: E402
from memhygiene.runner import _make_nli  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    cache = Path(cfg.get("cache_dir", "cache"))
    out = Path(cfg.get("out", "results/natural"))
    out.mkdir(parents=True, exist_ok=True)
    emb = make_embedder(cfg.get("embedder", {}), cache_dir=cache)
    hcfg = HygieneConfig(**cfg.get("hygiene", {}))
    nli = _make_nli(cfg, cache)
    llm = Claude(**cfg.get("llm", {}), cache_dir=cache)
    rows, dump = [], []
    for cname, ckw in cfg["corpora"].items():
        corpus = load_corpus(cname, **(ckw or {}))
        gate = gate_from_config(cfg.get("gate"), cname)
        for store in corpus.stores:
            mems = write_memories(llm, store, cfg.get("max_sessions"))
            if len(mems) < 2:
                continue
            V = encode_docs(emb, [m.text for m in mems])
            labels = annotate(llm, mems, V)
            full = run_hygiene(mems, V, hcfg, nli)
            gated, _ = gate.apply(full, store_features(mems, V, hcfg.tau_dup, hcfg.tau_sup))
            junk = np.array([l != "valid" for l in labels])
            for mode, res in (("full", full), ("gated", gated)):
                f = res.scores >= 0.5
                rows.append(dict(corpus=cname, store=store.id, hygiene=mode, n=len(mems), n_junk=int(junk.sum()),
                                 false_invalidation=float((f & ~junk).sum() / max(1, (~junk).sum())),
                                 junk_recall=float((f & junk).sum() / max(1, junk.sum())),
                                 flag_precision=float((f & junk).sum() / max(1, f.sum()))))
            for m, l, s, r in zip(mems, labels, full.scores, full.reasons):
                dump.append(dict(store=store.id, id=m.id, text=m.text, timestamp=m.timestamp, label=l,
                                 hygiene_score=float(s), hygiene_reason=r))
            print(f"{store.id}: {len(mems)} memories, {int(junk.sum())} labelled junk")
    pd.DataFrame(dump).to_csv(out / "natural_memories.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(out / "natural_hygiene.csv", index=False)
    print(df.groupby(["corpus", "hygiene"])[["false_invalidation", "junk_recall", "flag_precision"]].mean())
    labels = pd.DataFrame(dump).label.value_counts().to_dict()
    (out / "label_counts.json").write_text(json.dumps(labels, indent=2))
