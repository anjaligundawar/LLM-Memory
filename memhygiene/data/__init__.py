"""Dataset registry: load_corpus(name, **kwargs) returns a Corpus."""
from __future__ import annotations

from pathlib import Path

from . import clutrr, locomo, loghub, longmemeval, mab_conflict
from .download import DOWNLOADERS, RAW

CORPORA = ["locomo", "longmemeval", "mab_conflict", "clutrr", "loghub"]


def load_corpus(name: str, download: bool = True, **kw):
    if download:
        DOWNLOADERS[name]()
    if name == "locomo":
        return locomo.load(RAW / "locomo" / "locomo10.json", **kw)
    if name == "longmemeval":
        variant = kw.pop("variant", "s")
        fname = {"s": "longmemeval_s_cleaned.json", "oracle": "longmemeval_oracle.json",
                 "m": "longmemeval_m_cleaned.json"}[variant]
        return longmemeval.load(RAW / "longmemeval" / fname, **kw)
    if name == "mab_conflict":
        return mab_conflict.load(RAW / "mab" / "data" / "Conflict_Resolution-00000-of-00001.parquet", **kw)
    if name == "clutrr":
        return clutrr.load(RAW / "clutrr", **kw)
    if name == "loghub":
        return loghub.load(RAW / "loghub", **kw)
    raise KeyError(f"unknown corpus {name!r}; choose from {CORPORA}")


__all__ = ["CORPORA", "load_corpus", "RAW", "Path"]
