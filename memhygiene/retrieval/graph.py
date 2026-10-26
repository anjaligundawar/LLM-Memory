"""Graph memory: a bipartite graph between memory units and the keys they
mention (named entities, identifiers, numbers/dates and salient content
terms). Retrieval runs Personalized PageRank seeded at the query's keys and
ranks memory nodes by their stationary mass (HippoRAG-style, Gutierrez et al.
2024), so memories two hops away through a shared entity can surface.

Entity extraction uses spaCy (en_core_web_sm) when installed and falls back to
the regex extractor in memhygiene.text.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import sparse

from ..text import STOPWORDS, entities, tokenize
from .base import Retriever


@lru_cache(maxsize=1)
def _spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm", disable=["lemmatizer", "parser"])
    except Exception:
        return None


ENT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC", "NORP", "PRODUCT", "EVENT", "WORK_OF_ART",
              "DATE", "TIME", "MONEY", "QUANTITY", "CARDINAL", "LANGUAGE"}


_KEY_CACHE: dict = {}
_KEY_CACHE_MAX = 500_000


def extract_keys(texts, use_spacy: bool = True, min_term_len: int = 3):
    """Per text: set of 'e:<entity>' and 't:<term>' keys (memoised per text,
    since the same genuine memories are re-indexed in every grid cell)."""
    nlp = _spacy() if use_spacy else None
    tag = "spacy" if nlp is not None else "regex"
    todo = list(dict.fromkeys(t for t in texts if (tag, t) not in _KEY_CACHE))
    if todo:
        # make room BEFORE storing this batch, so the keys just computed are kept
        if len(_KEY_CACHE) + len(todo) > _KEY_CACHE_MAX:
            _KEY_CACHE.clear()
            todo = list(dict.fromkeys(texts))
        if nlp is not None:
            ents = [{e.text.lower() for e in doc.ents if e.label_ in ENT_LABELS}
                    for doc in nlp.pipe(todo, batch_size=256)]
        else:
            ents = [{e.lower() for e in entities(t)} for t in todo]
        for t, es in zip(todo, ents):
            keys = {f"e:{e}" for e in es}
            keys |= {f"t:{w}" for w in tokenize(t) if w not in STOPWORDS and len(w) >= min_term_len
                     and not w.isdigit()}
            _KEY_CACHE[(tag, t)] = frozenset(keys)
    return [_KEY_CACHE[(tag, t)] for t in texts]


class GraphRetriever(Retriever):
    name = "graph"

    def __init__(self, damping: float = 0.5, iters: int = 30, use_spacy: bool = True,
                 entity_weight: float = 2.0, max_key_df: float = 0.2):
        self.damping, self.iters = damping, iters
        self.use_spacy, self.entity_weight, self.max_key_df = use_spacy, entity_weight, max_key_df

    def _build(self, vectors):
        texts = [u.text for u in self.units]
        keys = extract_keys(texts, self.use_spacy)
        n = len(texts)
        df: dict = {}
        for ks in keys:
            for k in ks:
                df[k] = df.get(k, 0) + 1
        # very common keys (speaker names, "user", template words) connect everything; drop them
        cap = max(2, int(self.max_key_df * n))
        self.key_index = {k: i for i, k in enumerate(k for k, c in df.items() if c <= cap)}
        rows, cols, vals = [], [], []
        for i, ks in enumerate(keys):
            for k in ks:
                j = self.key_index.get(k)
                if j is not None:
                    w = self.entity_weight if k.startswith("e:") else 1.0
                    rows.append(i); cols.append(j); vals.append(w * np.log(1 + n / df[k]))
        m = max(1, len(self.key_index))
        B = sparse.csr_matrix((vals, (rows, cols)), shape=(n, m), dtype=np.float64)
        # column-stochastic transitions unit->key and key->unit
        self.U2K = _row_normalise(B)            # n x m
        self.K2U = _row_normalise(B.T.tocsr())  # m x n
        self.n_keys = m

    def seeds(self, query: str) -> np.ndarray:
        qkeys = extract_keys([query], self.use_spacy)[0]
        p = np.zeros(self.n_keys)
        for k in qkeys:
            j = self.key_index.get(k)
            if j is not None:
                p[j] += self.entity_weight if k.startswith("e:") else 1.0
        return p

    def scores(self, query, qvec):
        p_keys = self.seeds(query)
        if p_keys.sum() == 0:
            return np.zeros(len(self.units))
        p_keys /= p_keys.sum()
        # random walk alternating key -> unit -> key, restarting at the seed keys
        k_mass = p_keys.copy()
        u_total = np.zeros(len(self.units))
        for _ in range(self.iters):
            u_mass = self.K2U.T @ k_mass
            u_total += u_mass
            k_mass = (1 - self.damping) * p_keys + self.damping * (self.U2K.T @ u_mass)
        return u_total / self.iters


def _row_normalise(M: sparse.csr_matrix) -> sparse.csr_matrix:
    s = np.asarray(M.sum(axis=1)).ravel()
    s[s == 0] = 1.0
    return sparse.diags(1.0 / s) @ M
