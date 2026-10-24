"""LongMemEval (cleaned, Sept 2025 release): 500 questions, each with its own
chat history ("haystack") of ~50 sessions between a user and an assistant.

Store = one question's haystack. Unit = one chat turn. Gold = turns marked
has_answer. Abstention questions (id ends with _abs) have no gold and are
dropped. The knowledge-update questions are the natural "stale memory" case.
"""
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

from ..schema import Corpus, MemoryUnit, Query, Store


def _parse_dt(s: str) -> float:
    # e.g. "2023/05/20 (Sat) 02:21"
    s = s.split("(")[0].strip() + " " + s.split(")")[-1].strip()
    return datetime.strptime(s, "%Y/%m/%d %H:%M").timestamp()


def load(path: Path, max_stores: int | None = None, include_assistant: bool = True,
         max_chars: int = 2000, seed: int = 0) -> Corpus:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data = [d for d in data if not d["question_id"].endswith("_abs")]
    if max_stores and max_stores < len(data):
        # stratified by question type so every type stays represented
        rng = random.Random(seed)
        by_type: dict = {}
        for d in data:
            by_type.setdefault(d["question_type"], []).append(d)
        picked = []
        per = max(1, max_stores // len(by_type))
        for items in by_type.values():
            picked += rng.sample(items, min(per, len(items)))
        data = picked[:max_stores]
    stores = []
    for d in data:
        sid = f"lme/{d['question_id']}"
        units, gold = [], []
        for si, (sess_id, date, sess) in enumerate(zip(d["haystack_session_ids"], d["haystack_dates"],
                                                       d["haystack_sessions"])):
            ts0 = _parse_dt(date)
            for ti, turn in enumerate(sess):
                if turn["role"] != "user" and not include_assistant:
                    continue
                uid = f"{sid}/{sess_id}/{ti}"
                units.append(MemoryUnit(id=uid, text=f"{turn['role']}: {turn['content'][:max_chars]}",
                                        timestamp=ts0 + ti, store_id=sid,
                                        meta={"session": sess_id, "role": turn["role"]}))
                if turn.get("has_answer"):
                    gold.append(uid)
        if not gold:
            continue
        q = Query(id=f"{sid}/q", store_id=sid, text=d["question"], gold_ids=gold,
                  answer=str(d["answer"]), qtype=d["question_type"],
                  timestamp=_parse_dt(d["question_date"]))
        stores.append(Store(id=sid, units=units, queries=[q]))
    return Corpus(name="longmemeval", stores=stores, meta={"unit": "chat turn"})
