"""Risk-aware ("gated") hygiene: our proposed method.

Always-on hygiene fails on templated memories (logs, kinship facts), where
genuine memories look like duplicates or updates of one another. The gate
decides, per store and per component, whether running that component is
safe, using only label-free store features:

  1. predicted harm: for each component, a ridge model maps store features
     plus that component's own flag rate to the expected share of genuine
     memories it would wrongly invalidate (fitted on other corpora,
     leave-one-corpus-out; see scripts/fit_gate.py).
     Without a fitted model a transparent rule is used: a detector that
     wants to invalidate more than `max_flag_rate` of a store is treated as
     miscalibrated for that store (no realistic memory is mostly junk of one
     kind), so it is switched off.
  2. the component is enabled only if predicted harm <= epsilon
  3. budget: an enabled component may invalidate at most `budget` of the
     store; beyond that only its highest-scoring flags are kept
  4. scores are applied softly (down-weighting, never deletion)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..features import FEATURES
from .core import COMPONENTS, HygieneResult, combine


@dataclass
class GateModel:
    coef: Dict[str, List[float]] = field(default_factory=dict)       # component -> weights
    intercept: Dict[str, float] = field(default_factory=dict)
    mean: List[float] = field(default_factory=list)
    std: List[float] = field(default_factory=list)

    def predict(self, feats: Dict[str, float], flag_rates: Dict[str, float]) -> Dict[str, float]:
        out = {}
        for c, w in self.coef.items():
            raw = np.array([feats[f] for f in FEATURES] + [flag_rates.get(c, 0.0)])
            x = (raw - np.array(self.mean)) / np.array(self.std)
            out[c] = float(np.clip(x @ np.array(w) + self.intercept[c], 0, 1))
        return out

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GateModel":
        return cls(**json.loads(Path(path).read_text()))


def rule_predict(flag_rates: Dict[str, float], max_flag_rate: float) -> Dict[str, float]:
    """Fallback harm estimate before any model is fitted: 0 when a component
    flags a plausible share of the store, 1 when it flags more than
    max_flag_rate (then it is almost certainly hitting genuine memories)."""
    return {c: 0.0 if r <= max_flag_rate else 1.0 for c, r in flag_rates.items()}


def flag_rates(res: HygieneResult, threshold: float = 0.5) -> Dict[str, float]:
    return {c: float((s >= threshold).mean()) if len(s) else 0.0 for c, s in res.by_component.items()}


@dataclass
class Gate:
    epsilon: float = 0.05
    budget: float = 0.5
    max_flag_rate: float = 0.5
    model: Optional[GateModel] = None

    def decide(self, feats: Dict[str, float], flag_rates: Dict[str, float]) -> Dict[str, bool]:
        harm = self.model.predict(feats, flag_rates) if self.model else rule_predict(flag_rates, self.max_flag_rate)
        return {c: harm.get(c, 1.0) <= self.epsilon for c in flag_rates}

    def apply(self, full: HygieneResult, feats: Dict[str, float]) -> tuple:
        comps = list(full.by_component)
        on = self.decide(feats, flag_rates(full))
        gated = {}
        for c in comps:
            s = full.by_component[c].copy() if on[c] else np.zeros_like(full.by_component[c])
            n_max = int(self.budget * len(s))
            if on[c] and (s > 0).sum() > n_max:
                keep = np.argsort(-s, kind="stable")[:n_max]
                trimmed = np.zeros_like(s)
                trimmed[keep] = s[keep]
                s = trimmed
            gated[c] = s
        return combine(gated, comps), on
