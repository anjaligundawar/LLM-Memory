"""NLI contradiction scorer (cross-encoder) with a pair cache."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


class NLIScorer:
    def __init__(self, model: str = "cross-encoder/nli-MiniLM2-L6-H768", cache_dir: Path | None = None,
                 device: str | None = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder
        self.name = model
        self.model = CrossEncoder(model, device=device)
        id2label = {int(k): v.lower() for k, v in self.model.model.config.id2label.items()}
        self.contra_idx = next(i for i, l in id2label.items() if l.startswith("contra"))
        self.batch_size = batch_size
        self.db = None
        if cache_dir:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            from ..embed import connect_cache
            self.db = connect_cache(Path(cache_dir) / f"nli_{model.replace('/', '__')}.sqlite")
            self.db.execute("CREATE TABLE IF NOT EXISTS p (k TEXT PRIMARY KEY, v REAL)")

    @staticmethod
    def _k(a: str, b: str) -> str:
        return hashlib.sha1(f"{a}\x00{b}".encode("utf-8")).hexdigest()

    def contradiction(self, pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        """P(contradiction) for each (premise, hypothesis) pair."""
        keys = [self._k(a, b) for a, b in pairs]
        have = {}
        if self.db is not None and keys:
            for i in range(0, len(keys), 900):
                chunk = keys[i:i + 900]
                q = f"SELECT k, v FROM p WHERE k IN ({','.join('?' * len(chunk))})"
                have.update(dict(self.db.execute(q, chunk)))
        todo = [(k, p) for k, p in zip(keys, pairs) if k not in have]
        if todo:
            logits = self.model.predict([p for _, p in todo], batch_size=self.batch_size,
                                        show_progress_bar=False, apply_softmax=True)
            probs = np.asarray(logits)[:, self.contra_idx]
            new = {k: float(v) for (k, _), v in zip(todo, probs)}
            have.update(new)
            if self.db is not None:
                self.db.executemany("INSERT OR REPLACE INTO p VALUES (?, ?)", list(new.items()))
                self.db.commit()
        return np.array([have[k] for k in keys], dtype=np.float32)


class OverlapContradiction:
    """Label-free stand-in used in tests: flags pairs that share most words but
    differ in a negation or one value slot. Not used for reported results."""
    name = "overlap"

    def contradiction(self, pairs):
        from ..text import slots, tokenize
        out = []
        for a, b in pairs:
            ta, tb = set(tokenize(a)), set(tokenize(b))
            jac = len(ta & tb) / max(1, len(ta | tb))
            neg = ("not" in ta) != ("not" in tb)
            diff = bool(slots(a) ^ slots(b))
            out.append(jac if (neg or diff) and jac > 0.6 else 0.0)
        return np.array(out, dtype=np.float32)
