"""Turning invalidation scores into retrieval behaviour.

  hard  delete every unit with score >= 0.5 before indexing (what most
        memory systems do when they "UPDATE"/"DELETE" a memory)
  soft  keep everything, but down-weight: final = base * (1 - lam * score),
        where base is the retriever score min-max normalised over the top
        `pool` candidates (soft invalidation; nothing is lost for good)
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np


def hard_keep_mask(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return scores < threshold


def soft_rerank(ranked: List[Tuple[str, float]], inv: dict, lam: float = 0.9, k: int = 10):
    if not ranked:
        return []
    base = np.array([s for _, s in ranked], dtype=np.float64)
    lo, hi = base.min(), base.max()
    norm = (base - lo) / (hi - lo) if hi > lo else np.ones_like(base)
    adj = [(uid, float(b * (1 - lam * inv.get(uid, 0.0)))) for (uid, _), b in zip(ranked, norm)]
    adj.sort(key=lambda x: -x[1])
    return adj[:k]
