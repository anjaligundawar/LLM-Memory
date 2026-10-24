"""Memory hygiene: detectors that assign each memory an invalidation score.

Components (each can be switched on/off and ablated):

  dedup         near-duplicate pairs (cosine >= tau_dup): the LATER copy is
                invalidated, the earliest is kept
  supersession  similar pairs (tau_sup <= cosine < tau_dup) that share their
                wording but differ in a value slot (number, date, name, id):
                the OLDER one is treated as an outdated version
  contradiction NLI cross-encoder on nearest-neighbour pairs (cosine >=
                tau_cand); if P(contradiction) >= tau_nli the OLDER memory is
                invalidated ("newest wins", as in Mem0 / Zep-style updates)
  transient     low-information small talk (informativeness < tau_info)
  isolation     (off by default) memories unlike anything else in the store

The result is a score in [0, 1] per unit plus the reason. How the scores are
used (hard deletion, soft down-weighting, gating) lives in apply.py / gate.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..schema import MemoryUnit
from ..text import informativeness, slots, tokenize

COMPONENTS = ["dedup", "supersession", "contradiction", "transient", "isolation"]


@dataclass
class HygieneConfig:
    dedup: bool = True
    supersession: bool = True
    contradiction: bool = True
    transient: bool = True
    isolation: bool = False
    tau_dup: float = 0.92
    tau_sup: float = 0.75
    sup_overlap: float = 0.5
    tau_cand: float = 0.65
    nli_neighbours: int = 3
    tau_nli: float = 0.5
    tau_info: float = 0.2
    tau_iso: float = 0.2

    def enabled(self) -> List[str]:
        return [c for c in COMPONENTS if getattr(self, c)]


@dataclass
class HygieneResult:
    scores: np.ndarray                         # combined invalidation score per unit
    reasons: List[Optional[str]]
    by_component: Dict[str, np.ndarray] = field(default_factory=dict)

    def flagged(self, threshold: float = 0.5) -> np.ndarray:
        return self.scores >= threshold


def _later(units: Sequence[MemoryUnit], i: int, j: int) -> int:
    ti, tj = units[i].timestamp, units[j].timestamp
    if ti == tj:
        return max(i, j)
    return i if ti > tj else j


def run_hygiene(units: Sequence[MemoryUnit], vectors: np.ndarray, cfg: HygieneConfig,
                nli=None, components: Optional[Sequence[str]] = None) -> HygieneResult:
    """Score every unit. `components` overrides cfg (used to compute every
    component once and gate afterwards)."""
    n = len(units)
    comps = list(components) if components is not None else cfg.enabled()
    out = {c: np.zeros(n, dtype=np.float32) for c in comps}
    if n == 0:
        return HygieneResult(np.zeros(0), [], out)
    V = np.asarray(vectors, dtype=np.float32)
    S = V @ V.T
    np.fill_diagonal(S, -1.0)
    texts = [u.text for u in units]

    if "dedup" in comps:
        ii, jj = np.where(np.triu(S >= cfg.tau_dup, 1))
        for i, j in zip(ii, jj):
            out["dedup"][_later(units, i, j)] = 1.0

    if "supersession" in comps:
        ii, jj = np.where(np.triu((S >= cfg.tau_sup) & (S < cfg.tau_dup), 1))
        slot_cache: Dict[int, set] = {}
        tok_cache: Dict[int, set] = {}
        for i, j in zip(ii, jj):
            for x in (i, j):
                if x not in slot_cache:
                    slot_cache[x] = slots(texts[x])
                    tok_cache[x] = set(tokenize(texts[x])) - slot_cache[x]
            if not (slot_cache[i] ^ slot_cache[j]):
                continue
            a, b = tok_cache[i], tok_cache[j]
            if len(a & b) / max(1, len(a | b)) < cfg.sup_overlap:
                continue
            older = j if _later(units, i, j) == i else i
            out["supersession"][older] = max(out["supersession"][older], float(S[i, j]))

    if "contradiction" in comps and nli is not None:
        k = min(cfg.nli_neighbours, n - 1)
        pairs = set()
        if k > 0:
            nbrs = np.argpartition(-S, k - 1, axis=1)[:, :k]
            for i in range(n):
                for j in nbrs[i]:
                    if S[i, j] >= cfg.tau_cand and S[i, j] < cfg.tau_dup:
                        pairs.add((min(i, j), max(i, j)))
        pairs = sorted(pairs)
        if pairs:
            both = nli.contradiction([(texts[i], texts[j]) for i, j in pairs] +
                                     [(texts[j], texts[i]) for i, j in pairs])
            p = np.maximum(both[:len(pairs)], both[len(pairs):])
            for (i, j), pc in zip(pairs, p):
                if pc >= cfg.tau_nli:
                    older = j if _later(units, i, j) == i else i
                    out["contradiction"][older] = max(out["contradiction"][older], float(pc))

    if "transient" in comps:
        info = np.array([informativeness(t) for t in texts], dtype=np.float32)
        out["transient"] = np.where(info < cfg.tau_info, 1.0 - info / max(cfg.tau_info, 1e-6), 0.0)

    if "isolation" in comps:
        nn = S.max(axis=1)
        out["isolation"] = np.where(nn < cfg.tau_iso, 1.0, 0.0).astype(np.float32)

    return combine(out, comps)


def combine(by_component: Dict[str, np.ndarray], comps: Sequence[str]) -> HygieneResult:
    n = len(next(iter(by_component.values()))) if by_component else 0
    scores = np.zeros(n, dtype=np.float32)
    reasons: List[Optional[str]] = [None] * n
    for c in comps:
        s = by_component.get(c)
        if s is None:
            continue
        better = s > scores
        for i in np.where(better)[0]:
            reasons[i] = c
        scores = np.maximum(scores, s)
    return HygieneResult(scores, reasons, {c: by_component[c] for c in comps if c in by_component})
