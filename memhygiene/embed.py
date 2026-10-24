"""Text embedders with an on-disk cache.

Every experiment re-embeds the same genuine memories under many pollution
settings, so vectors are cached by (model, text) in a small SQLite file.
"""
from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def connect_cache(path) -> sqlite3.Connection:
    """SQLite connection that several runs can share: WAL journal and a long
    lock timeout, so a parallel run waits instead of failing."""
    db = sqlite3.connect(str(path), timeout=120, check_same_thread=False)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    return db


def _key(model: str, text: str) -> str:
    return hashlib.sha1(f"{model}\x00{text}".encode("utf-8")).hexdigest()


class VectorCache:
    def __init__(self, path: Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = connect_cache(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS v (k TEXT PRIMARY KEY, d BLOB)")
        self.lock = threading.Lock()

    def get_many(self, keys: Sequence[str]) -> Dict[str, np.ndarray]:
        out = {}
        for i in range(0, len(keys), 900):
            chunk = list(keys[i:i + 900])
            q = f"SELECT k, d FROM v WHERE k IN ({','.join('?' * len(chunk))})"
            for k, d in self.db.execute(q, chunk):
                out[k] = np.frombuffer(d, dtype=np.float32)
        return out

    def put_many(self, items: Dict[str, np.ndarray]) -> None:
        with self.lock:
            self.db.executemany("INSERT OR REPLACE INTO v VALUES (?, ?)",
                                [(k, v.astype(np.float32).tobytes()) for k, v in items.items()])
            self.db.commit()


class Embedder:
    """Base class: subclasses implement _encode(texts) -> unit-norm float32 matrix."""
    name = "base"

    def __init__(self, cache_dir: Path | None = None):
        self.cache = VectorCache(Path(cache_dir) / f"emb_{self.slug()}.sqlite") if cache_dir else None
        self.mem: Dict[str, np.ndarray] = {}

    def slug(self) -> str:
        return self.name.replace("/", "__")

    def _encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        keys = [_key(self.name, t) for t in texts]
        missing = [k for k in dict.fromkeys(keys) if k not in self.mem]
        if missing and self.cache:
            self.mem.update(self.cache.get_many(missing))
            missing = [k for k in missing if k not in self.mem]
        if missing:
            by_key = {k: t for k, t in zip(keys, texts)}
            todo = [by_key[k] for k in missing]
            vecs = self._encode(todo).astype(np.float32)
            new = dict(zip(missing, vecs))
            self.mem.update(new)
            if self.cache:
                self.cache.put_many(new)
        return np.stack([self.mem[k] for k in keys]) if keys else np.zeros((0, self.dim()), np.float32)

    def dim(self) -> int:
        return int(self._encode(["x"]).shape[1])


class SentenceTransformerEmbedder(Embedder):
    """Any sentence-transformers model, e.g. all-MiniLM-L6-v2, BAAI/bge-small-en-v1.5,
    intfloat/e5-small-v2 (e5 needs the query:/passage: prefixes, set via prefixes)."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", cache_dir: Path | None = None,
                 device: str | None = None, batch_size: int = 64, query_prefix: str = "", doc_prefix: str = ""):
        self.name = model
        super().__init__(cache_dir)
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model, device=device)
        self.batch_size = batch_size
        self.query_prefix, self.doc_prefix = query_prefix, doc_prefix

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                 show_progress_bar=len(texts) > 2000, convert_to_numpy=True)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode([self.query_prefix + t for t in texts])

    def encode_docs(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode([self.doc_prefix + t for t in texts])

    def dim(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())


class HashingEmbedder(Embedder):
    """Offline lexical stand-in (word + char n-gram hashing). Only for tests and
    smoke runs on machines without the neural model; never for reported results."""
    name = "hashing"

    def __init__(self, n_features: int = 2 ** 14, cache_dir: Path | None = None):
        super().__init__(cache_dir)
        from sklearn.feature_extraction.text import HashingVectorizer
        self.vec = HashingVectorizer(n_features=n_features, analyzer="char_wb", ngram_range=(3, 4),
                                     alternate_sign=False, norm="l2")

    def _encode(self, texts: List[str]) -> np.ndarray:
        return self.vec.transform(texts).toarray().astype(np.float32)


def _encode_role(emb: Embedder, texts: Sequence[str], role: str) -> np.ndarray:
    fn = getattr(emb, f"encode_{role}", None)
    return fn(texts) if fn else emb.encode(texts)


def encode_docs(emb: Embedder, texts: Sequence[str]) -> np.ndarray:
    return _encode_role(emb, texts, "docs")


def encode_queries(emb: Embedder, texts: Sequence[str]) -> np.ndarray:
    return _encode_role(emb, texts, "queries")


def set_threads(n: int | str | None = None) -> int:
    """Use all CPU cores for torch (embeddings, NLI) unless told otherwise.
    When running several grids in parallel, give each a share, e.g. threads: 4.
    Also settable with the MEMHYG_THREADS environment variable."""
    import os
    n = os.environ.get("MEMHYG_THREADS", n)
    n = (os.cpu_count() or 1) if n in (None, "auto") else int(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass
    return n


def make_embedder(cfg: dict, cache_dir: Path | None = None) -> Embedder:
    name = cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2")
    if name == "hashing":
        return HashingEmbedder(cache_dir=cache_dir)
    return SentenceTransformerEmbedder(name, cache_dir=cache_dir, device=cfg.get("device"),
                                       batch_size=cfg.get("batch_size", 64),
                                       query_prefix=cfg.get("query_prefix", ""),
                                       doc_prefix=cfg.get("doc_prefix", ""))
