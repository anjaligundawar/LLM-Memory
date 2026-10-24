"""LoCoMo: 10 very long two-person conversations (~300 turns, up to 35
sessions each) with ~2k QA pairs whose evidence is given as dialogue ids.

Store = one conversation. Unit = one dialogue turn. Gold = QA evidence turns.
Category 5 (adversarial, unanswerable) questions are dropped because they
have no evidence to retrieve.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from ..schema import Corpus, MemoryUnit, Query, Store

CATEGORY = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def _parse_dt(s: str) -> float:
    s = s.strip()
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y", "%I:%M %p on %B %d, %Y"):
        try:
            return datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    raise ValueError(f"unparsed LoCoMo date: {s!r}")


def load(path: Path, max_stores: int | None = None) -> Corpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    stores = []
    for conv in data[:max_stores]:
        sid = f"locomo/{conv['sample_id']}"
        c = conv["conversation"]
        units = []
        n_sessions = max(int(m.group(1)) for k in c for m in [re.fullmatch(r"session_(\d+)", k)] if m)
        last_ts = 0.0
        for n in range(1, n_sessions + 1):
            turns = c.get(f"session_{n}")
            if not turns:
                continue
            ts0 = _parse_dt(c[f"session_{n}_date_time"])
            for i, t in enumerate(turns):
                text = f"{t['speaker']}: {t.get('text', '')}"
                if t.get("blig_caption") or t.get("blip_caption"):
                    text += f" [shares a photo of {t.get('blip_caption') or t.get('blig_caption')}]"
                units.append(MemoryUnit(id=f"{sid}/{t['dia_id']}", text=text, timestamp=ts0 + i,
                                        store_id=sid, meta={"session": n}))
                last_ts = ts0 + i
        ids = {u.id for u in units}
        queries = []
        for qi, qa in enumerate(conv["qa"]):
            if qa.get("category") == 5:
                continue
            ev = []
            for e in qa.get("evidence", []):
                ev += [x.strip() for x in re.split(r"[;,]", e) if x.strip()]
            gold = [f"{sid}/{e}" for e in ev if f"{sid}/{e}" in ids]
            if not gold:
                continue
            queries.append(Query(id=f"{sid}/q{qi}", store_id=sid, text=qa["question"], gold_ids=gold,
                                 answer=str(qa.get("answer", "")), qtype=CATEGORY.get(qa.get("category")),
                                 timestamp=last_ts))
        stores.append(Store(id=sid, units=units, queries=queries))
    return Corpus(name="locomo", stores=stores, meta={"unit": "dialogue turn"})
