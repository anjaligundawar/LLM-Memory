"""CLUTRR: kinship stories with multi-hop relation questions.

One CLUTRR story has only 2-10 sentences, too few for retrieval to mean
anything, so a Store packs `stories_per_store` stories whose character names
do not overlap (an agent that remembers many families). Unit = one sentence.
Query = "How is B related to A?". Gold = the sentences that state each hop on
the story's proof chain. The many templated, near-identical sentences make
this the stress test for deduplication.
"""
from __future__ import annotations

import ast
import random
import re
from pathlib import Path

import pandas as pd

from ..schema import Corpus, MemoryUnit, Query, Store

SENT_RE = re.compile(r"(?<=[\.!?])\s+")


def _names(genders: str):
    return [g.split(":")[0].strip() for g in genders.split(",")]


def _story(r):
    names = _names(r["genders"])
    edges = ast.literal_eval(r["story_edges"])
    sents = [s.strip() for s in SENT_RE.split(r["story"]) if s.strip()]
    gold = set()
    for u_, v_ in edges:
        if u_ < len(names) and v_ < len(names):
            for i, s in enumerate(sents):
                if f"[{names[u_]}]" in s and f"[{names[v_]}]" in s:
                    gold.add(i)
    a, b = ast.literal_eval(r["query"])
    return names, sents, sorted(gold), (a, b), len(edges)


def load(root: Path, config: str = "gen_train234_test2to10", max_stores: int | None = None,
         stories_per_store: int = 8, min_hops: int = 2, max_hops: int = 10, seed: int = 0) -> Corpus:
    df = pd.read_csv(Path(root) / config / "test.csv")
    rows = df.to_dict("records")
    random.Random(seed).shuffle(rows)
    stories = []
    for r in rows:
        names, sents, gold, (a, b), hops = _story(r)
        if min_hops <= hops <= max_hops and gold:
            stories.append((r, names, sents, gold, a, b, hops))
    stores, pool = [], stories
    while pool and (max_stores is None or len(stores) < max_stores):
        used, picked, rest = set(), [], []
        for st in pool:
            if len(picked) < stories_per_store and not (set(st[1]) & used):
                picked.append(st)
                used |= set(st[1])
            else:
                rest.append(st)
        pool = rest
        if len(picked) < stories_per_store:
            break
        sid = f"clutrr/{len(stores)}"
        units, queries, t = [], [], 0.0
        for k, (r, names, sents, gold, a, b, hops) in enumerate(picked):
            ids = []
            for i, s in enumerate(sents):
                uid = f"{sid}/st{k}s{i}"
                units.append(MemoryUnit(id=uid, text=s.replace("[", "").replace("]", ""), timestamp=t,
                                        store_id=sid, meta={"story": k}))
                ids.append(uid)
                t += 1.0
            queries.append(Query(id=f"{sid}/q{k}", store_id=sid, text=f"How is {b} related to {a}?",
                                 gold_ids=[ids[i] for i in gold], answer=r["target_text"],
                                 qtype=f"{hops}-hop"))
        for q in queries:
            q.timestamp = t
        stores.append(Store(id=sid, units=units, queries=queries))
    return Corpus(name="clutrr", stores=stores, meta={"unit": "story sentence", "config": config})
