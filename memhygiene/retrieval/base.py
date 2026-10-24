from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..schema import MemoryUnit


class Retriever:
    name = "base"
    needs_vectors = False

    def index(self, units: Sequence[MemoryUnit], vectors: Optional[np.ndarray] = None) -> "Retriever":
        self.units = list(units)
        self.ids = [u.id for u in self.units]
        self._build(vectors)
        return self

    def _build(self, vectors):
        raise NotImplementedError

    def scores(self, query: str, qvec: Optional[np.ndarray]) -> np.ndarray:
        """One score per indexed unit, higher is better."""
        raise NotImplementedError

    def search(self, query: str, qvec: Optional[np.ndarray] = None, k: int = 10) -> List[Tuple[str, float]]:
        if not self.units:
            return []
        s = self.scores(query, qvec)
        k = min(k, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.lexsort((top, -s[top]))]   # score desc, then index for determinism
        return [(self.ids[i], float(s[i])) for i in top]
