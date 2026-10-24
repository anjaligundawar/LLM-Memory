"""Hybrid memory: reciprocal rank fusion of dense and BM25 rankings."""
from __future__ import annotations

import numpy as np

from .base import Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever


def rrf(*score_lists: np.ndarray, k: int = 60) -> np.ndarray:
    out = np.zeros_like(score_lists[0], dtype=np.float64)
    for s in score_lists:
        ranks = np.empty(len(s), dtype=np.int64)
        ranks[np.argsort(-s, kind="stable")] = np.arange(len(s))
        out += 1.0 / (k + ranks + 1)
    return out


class HybridRetriever(Retriever):
    name = "hybrid"
    needs_vectors = True

    def _build(self, vectors):
        self.dense = DenseRetriever().index(self.units, vectors)
        self.bm25 = BM25Retriever().index(self.units)

    def scores(self, query, qvec):
        return rrf(self.dense.scores(query, qvec), self.bm25.scores(query, None))
