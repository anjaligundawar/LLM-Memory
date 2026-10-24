"""Okapi BM25 (k1=1.5, b=0.75) over the shared tokenizer, vectorised with scipy."""
from __future__ import annotations

import numpy as np
from scipy import sparse

from ..text import STOPWORDS, tokenize
from .base import Retriever


def analyze(text: str):
    return [t for t in tokenize(text) if t not in STOPWORDS]


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def _build(self, vectors):
        vocab, rows, cols, vals = {}, [], [], []
        lens = []
        for i, u in enumerate(self.units):
            toks = analyze(u.text)
            lens.append(len(toks))
            counts = {}
            for t in toks:
                counts[t] = counts.get(t, 0) + 1
            for t, c in counts.items():
                j = vocab.setdefault(t, len(vocab))
                rows.append(i); cols.append(j); vals.append(c)
        self.vocab = vocab
        n = len(self.units)
        tf = sparse.csr_matrix((vals, (rows, cols)), shape=(n, max(1, len(vocab))), dtype=np.float32)
        df = np.bincount(cols, minlength=max(1, len(vocab))) if cols else np.zeros(1)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        lens = np.asarray(lens, dtype=np.float32)
        avg = lens.mean() if n else 1.0
        norm = self.k1 * (1 - self.b + self.b * lens / max(avg, 1e-9))
        tf = tf.tocoo()
        w = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.w = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def scores(self, query, qvec):
        idx = [self.vocab[t] for t in set(analyze(query)) if t in self.vocab]
        if not idx:
            return np.zeros(len(self.units), dtype=np.float32)
        return np.asarray(self.w[:, idx] @ self.idf[idx]).ravel()
