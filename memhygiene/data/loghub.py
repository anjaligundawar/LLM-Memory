"""Loghub: labelled 2k-line samples of real system logs (HDFS, Spark, BGL...).

Store = a window of consecutive lines from one system. Unit = one log line.
Queries are identifier lookups ("What happened to blk_123?"): gold = every
line in the window that mentions that identifier. Logs are dominated by
repeated templates that differ only in ids and numbers, which is exactly the
pattern that fools deduplication and supersession.
"""
from __future__ import annotations

import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..schema import Corpus, MemoryUnit, Query, Store
from ..text import IDENT_RE


def _ts(row, i: int) -> float:
    # timestamp columns differ per system; fall back to line order
    for date_col, time_col in (("Date", "Time"), ("Day", "Time")):
        if date_col in row and time_col in row:
            try:
                return datetime.strptime(f"{row[date_col]} {row[time_col]}"[:40], "%y%m%d %H%M%S").timestamp()
            except (ValueError, TypeError):
                pass
    return float(i)


def load(root: Path, window: int = 200, max_windows_per_system: int = 5, queries_per_store: int = 10,
         max_gold: int = 5, seed: int = 0, systems: list | None = None) -> Corpus:
    rng = random.Random(seed)
    stores = []
    for f in sorted(Path(root).glob("*_2k.log_structured.csv")):
        system = f.name.split("_2k")[0]
        if systems and system not in systems:
            continue
        df = pd.read_csv(f, dtype=str).fillna("")
        rows = df.to_dict("records")
        for w in range(0, min(len(rows), window * max_windows_per_system), window):
            chunk = rows[w:w + window]
            if len(chunk) < window // 2:
                break
            sid = f"loghub/{system}/w{w // window}"
            units = []
            for i, r in enumerate(chunk):
                level = r.get("Level", "")
                comp = r.get("Component", "")
                head = " ".join(x for x in (level, comp) if x)
                text = f"{head}: {r['Content']}" if head else r["Content"]
                units.append(MemoryUnit(id=f"{sid}/l{i}", text=text, timestamp=_ts(r, w + i), store_id=sid,
                                        meta={"event": r.get("EventId", "")}))
            # identifiers that occur in 1..max_gold lines of this window
            occ: dict = {}
            for u in units:
                for ident in {m.group(0) for m in IDENT_RE.finditer(u.text)}:
                    occ.setdefault(ident, []).append(u.id)
            cands = [k for k, v in occ.items() if 1 <= len(v) <= max_gold and len(k) >= 5]
            rng.shuffle(cands)
            queries, used = [], set()
            for ident in cands:
                key = frozenset(occ[ident])
                if key in used:
                    continue
                used.add(key)
                queries.append(Query(id=f"{sid}/q{len(queries)}", store_id=sid,
                                     text=f"What happened with {ident}?", gold_ids=list(occ[ident]),
                                     qtype=f"{system}", timestamp=units[-1].timestamp))
                if len(queries) >= queries_per_store:
                    break
            if queries:
                stores.append(Store(id=sid, units=units, queries=queries, meta={"system": system}))
    return Corpus(name="loghub", stores=stores, meta={"unit": "log line"})


def routine_lines(corpus: Corpus, top_templates: int = 3) -> list:
    """Lines of the most frequent event templates: the log analogue of
    small talk (heartbeats, routine INFO lines). Used as 'temporary' junk."""
    out = []
    for s in corpus.stores:
        freq = Counter(u.meta.get("event") for u in s.units)
        common = {e for e, _ in freq.most_common(top_templates)}
        out += [u for u in s.units if u.meta.get("event") in common]
    return out
