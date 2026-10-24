"""Cluster (store-level) paired bootstrap.

Queries from the same memory store are not independent (they share the same
polluted store), so we resample stores with replacement and keep all their
queries. With one query per store this reduces to the usual paired bootstrap.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _cluster_index(clusters: Sequence) -> list:
    groups: Dict = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    return [np.array(v) for v in groups.values()]


def cluster_bootstrap_mean(x: Sequence[float], clusters: Optional[Sequence] = None, B: int = 2000,
                           seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    groups = _cluster_index(clusters if clusters is not None else range(len(x)))
    sums = np.array([x[g].sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(B, len(groups)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi)}


def cluster_bootstrap_diff(a: Sequence[float], b: Sequence[float], clusters: Optional[Sequence] = None,
                           B: int = 2000, seed: int = 0, alpha: float = 0.05) -> Dict[str, float]:
    """Mean of (a - b) with a percentile CI and a two-sided bootstrap p-value."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    r = cluster_bootstrap_mean(d, clusters, B, seed, alpha)
    groups = _cluster_index(clusters if clusters is not None else range(len(d)))
    sums = np.array([d[g].sum() for g in groups])
    cnts = np.array([len(g) for g in groups])
    rng = np.random.default_rng(seed + 1)
    idx = rng.integers(0, len(groups), size=(B, len(groups)))
    boots = sums[idx].sum(1) / cnts[idx].sum(1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    r["p"] = float(min(1.0, p))
    return r
