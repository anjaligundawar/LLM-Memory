"""Per-query retrieval metrics against gold evidence ids."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence


def query_metrics(ranked_ids: Sequence[str], gold: Iterable[str], junk_ids: set,
                  ks: Sequence[int] = (1, 3, 5, 10)) -> Dict[str, float]:
    gold = set(gold)
    out: Dict[str, float] = {}
    rr = 0.0
    for i, uid in enumerate(ranked_ids, 1):
        if uid in gold:
            rr = 1.0 / i
            break
    out["mrr"] = rr
    for k in ks:
        top = list(ranked_ids[:k])
        hits = [1 if u in gold else 0 for u in top]
        out[f"recall@{k}"] = sum(hits) / len(gold) if gold else 0.0
        out[f"hit@{k}"] = 1.0 if any(hits) else 0.0
        dcg = sum(h / math.log2(i + 2) for i, h in enumerate(hits))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(k, len(gold))))
        out[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
        out[f"junk@{k}"] = sum(1 for u in top if u in junk_ids) / k
    return out


def audc(levels: List[float], values: List[float]) -> float:
    """Area under a degradation curve, normalised by the level range, so it is
    the average metric over contamination levels (higher = more robust)."""
    pts = sorted(zip(levels, values))
    if len(pts) < 2:
        return pts[0][1] if pts else float("nan")
    area = sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    return area / (pts[-1][0] - pts[0][0])
