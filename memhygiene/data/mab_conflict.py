"""MemoryAgentBench, Conflict_Resolution split (FactConsolidation).

Each example is a long numbered list of facts in which later facts overwrite
earlier ones, plus questions whose answer is the most recent value. This is a
corpus with *natural* stale memories: the store already contains outdated
facts before we inject anything.

Store = one example. Unit = one numbered fact (timestamp = its index).
Gold = the latest fact that contains the answer string (derived, not given).
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ..schema import Corpus, MemoryUnit, Query, Store

FACT_RE = re.compile(r"^\s*(\d+)\.\s+(.*\S)\s*$")


def load(path: Path, max_stores: int | None = None, max_queries_per_store: int | None = 200) -> Corpus:
    df = pd.read_parquet(path)
    stores = []
    for ei, row in enumerate(df.to_dict("records")[:max_stores]):
        sid = f"mab/{ei}"
        units = []
        for line in str(row["context"]).splitlines():
            m = FACT_RE.match(line)
            if m:
                idx = int(m.group(1))
                units.append(MemoryUnit(id=f"{sid}/f{idx}", text=m.group(2), timestamp=float(idx), store_id=sid))
        if not units:
            continue
        lowered = [(u, u.text.lower()) for u in units]
        queries = []
        for qi, (q, answers) in enumerate(zip(row["questions"], row["answers"])):
            answers = [a for a in (list(answers) if answers is not None else []) if str(a).strip()]
            hits = [u for u, t in lowered if any(str(a).lower() in t for a in answers)]
            if not hits:
                continue
            latest = max(hits, key=lambda u: u.timestamp)
            queries.append(Query(id=f"{sid}/q{qi}", store_id=sid, text=str(q), gold_ids=[latest.id],
                                 answer=str(answers[0]), qtype="conflict", timestamp=units[-1].timestamp + 1))
            if max_queries_per_store and len(queries) >= max_queries_per_store:
                break
        if queries:
            stores.append(Store(id=sid, units=units, queries=queries))
    return Corpus(name="mab_conflict", stores=stores, meta={"unit": "fact"})
