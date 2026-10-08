"""
vector_retriever.py  -- MODULE 1 (Anjali): top-k cosine similarity search.

    sim(q, v) = (q . v) / (||q|| * ||v||)

Brute force over all n memories = O(n * d) per query.
"""
import time
import numpy as np
from vector_store import VectorStore


def cosine_sim(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class VectorRetriever:
    def __init__(self, units, embedder=None):
        self.store = VectorStore(units, embedder)

    def retrieve(self, query, k=3):
        """returns ([(MemoryUnit, score), ...] top-k, latency_ms)"""
        t0 = time.perf_counter()
        q_vec = self.store.embedder.encode([query])[0]
        scores = [(u, cosine_sim(q_vec, row))
                  for u, row in zip(self.store.units, self.store.matrix)]
        scores.sort(key=lambda x: x[1], reverse=True)
        latency_ms = (time.perf_counter() - t0) * 1000
        return scores[:k], latency_ms
