"""Dense vector memory: cosine similarity over unit-norm sentence embeddings
(exact search; stores here are at most a few thousand units)."""
from __future__ import annotations

import numpy as np

from .base import Retriever


class DenseRetriever(Retriever):
    name = "dense"
    needs_vectors = True

    def _build(self, vectors):
        if vectors is None:
            raise ValueError("DenseRetriever needs document vectors")
        self.mat = np.asarray(vectors, dtype=np.float32)

    def scores(self, query, qvec):
        return self.mat @ np.asarray(qvec, dtype=np.float32)
