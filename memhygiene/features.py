"""Label-free store features that predict when hygiene will hurt.

All features are computed on the (possibly polluted) store as an agent would
see it: no junk labels are used. They describe how templated, value-dense and
chatty the memory is.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .schema import MemoryUnit
from .text import NUMBER_RE, IDENT_RE, informativeness, tokenize

FEATURES = ["nn_sim_mean", "dup_pair_rate", "template_rate", "slot_density", "info_mean",
            "len_mean", "type_token_ratio"]


def store_features(units: Sequence[MemoryUnit], vectors: np.ndarray, tau_dup: float = 0.92,
                   tau_sup: float = 0.75) -> Dict[str, float]:
    n = len(units)
    if n < 2:
        return {f: 0.0 for f in FEATURES}
    V = np.asarray(vectors, dtype=np.float32)
    S = V @ V.T
    np.fill_diagonal(S, -1.0)
    nn = S.max(axis=1)
    toks = [tokenize(u.text) for u in units]
    all_toks = [t for ts in toks for t in ts]
    slot_counts = [len(NUMBER_RE.findall(u.text)) + len(IDENT_RE.findall(u.text)) for u in units]
    return {
        "nn_sim_mean": float(nn.mean()),
        "dup_pair_rate": float((nn >= tau_dup).mean()),
        "template_rate": float((nn >= tau_sup).mean()),
        "slot_density": float(np.mean([c / max(1, len(t)) for c, t in zip(slot_counts, toks)])),
        "info_mean": float(np.mean([informativeness(u.text) for u in units])),
        "len_mean": float(np.mean([len(t) for t in toks])),
        "type_token_ratio": float(len(set(all_toks)) / max(1, len(all_toks))),
    }
