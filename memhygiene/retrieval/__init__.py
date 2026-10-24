"""Retrievers share one interface:

    r = Retriever(...); r.index(units, doc_vectors); r.search(query_text, query_vector, k)

and return [(unit_id, score), ...] best first. Indexing is per polluted store.
"""
from .base import Retriever
from .bm25 import BM25Retriever
from .dense import DenseRetriever
from .graph import GraphRetriever
from .hybrid import HybridRetriever


def make_retriever(name: str, **kw) -> Retriever:
    if name == "dense":
        return DenseRetriever()
    if name == "bm25":
        return BM25Retriever()
    if name == "hybrid":
        return HybridRetriever()
    if name == "graph":
        return GraphRetriever(**kw)
    raise KeyError(name)


__all__ = ["Retriever", "DenseRetriever", "BM25Retriever", "HybridRetriever", "GraphRetriever",
           "make_retriever"]
