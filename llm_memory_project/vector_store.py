"""
vector_store.py  -- MODULE 1 (Anjali): the vector index.

Stores one vector per MemoryUnit (rows = memories, columns = dimensions)
and fills MemoryUnit.embedding in the shared schema.

Final review: replace the numpy matrix with FAISS or Chroma (HNSW index)
so search is fast on thousands of memories.
"""
import numpy as np
from embedder import TfidfEmbedder


class VectorStore:
    def __init__(self, units, embedder=None):
        self.units = list(units)
        self.embedder = embedder or TfidfEmbedder()
        texts = [u.text for u in self.units]
        self.embedder.fit(texts)
        self.matrix = np.asarray(self.embedder.encode(texts), dtype=float)
        for u, row in zip(self.units, self.matrix):
            u.embedding = row.tolist()          # Module 1 writes this field

    def size(self):
        return self.matrix.shape            # (number of memories, vector length)
