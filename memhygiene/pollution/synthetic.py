"""Synthetic junk-memory injection, generated from the corpus itself.

Five junk types (Section 3 of the paper):

  irrelevant     a memory that belongs to a different store (another user,
                 conversation or system), dropped in at a random time
  duplicate      the same fact written again later: an exact copy or a light
                 re-wording (filler prefix, dropped words, case/punctuation)
  stale          an OLDER version of a genuine fact with one value changed;
                 the genuine memory is the update that superseded it
  contradictory  a NEWER claim that conflicts with a genuine fact (one value
                 changed or negated); recency alone would wrongly prefer it
  temporary      transient small talk or routine state that should never
                 have been persisted (taken from low-information units of
                 other stores, or routine log templates)

Contamination level L means round(L * n_genuine) junk units are added, so
L = 0.75 adds three junk memories for every four genuine ones. A fraction
`targeted` of duplicate/stale/contradictory junk is derived from gold
evidence units, the rest from random genuine units; this controls how often
junk competes directly with the evidence a query needs.
"""
from __future__ import annotations

import random
import re
from typing import Dict, List, Optional, Sequence

from ..schema import Corpus, MemoryUnit, Store
from ..text import (MONTHS, NON_ENTITY_CAPS, NUMBER_RE, CAP_SPAN_RE, IDENT_RE, YEAR_RE,
                    informativeness, tokenize)

JUNK_TYPES = ["irrelevant", "duplicate", "stale", "contradictory", "temporary"]

FILLERS = ["As I mentioned, ", "Just to repeat, ", "Like I said before, ", "Reminder: ", "Again, ",
           "To recap, ", "Noting again that "]
AUX_NEG = [(r"\b(is|are|was|were|has|have|had|can|will|did|does|do|should|would)\b", r"\1 not")]
TRANSIENT_FALLBACK = [
    "I'm so tired right now, talk later.", "brb, grabbing coffee.", "It's raining here at the moment.",
    "Ugh, my internet is slow today.", "Hold on, someone's at the door.", "lol okay",
    "Just woke up, give me a sec.", "I'm in a meeting, will reply soon.", "Feeling a bit hungry now.",
    "Thanks! Sounds good.", "haha yes", "Ok cool, see you.",
]


class Polluter:
    def __init__(self, corpus: Corpus, targeted: float = 0.5, seed: int = 0,
                 transient_pool: Optional[Sequence[MemoryUnit]] = None, transient_max_info: float = 0.25):
        self.corpus = corpus
        self.targeted = targeted
        self.seed = seed
        all_units = [u for s in corpus.stores for u in s.units]
        if transient_pool is None:
            transient_pool = [u for u in all_units if informativeness(u.text) <= transient_max_info]
        self.transient_pool = list(transient_pool)
        self.by_store = {s.id: s for s in corpus.stores}
        # entity vocabulary per store for value substitution
        self.ent_vocab = {s.id: _cap_vocab(s.units) for s in corpus.stores}
        self.global_ents = sorted({e for v in self.ent_vocab.values() for e in v})

    # ------------------------------------------------------------------ public
    def pollute(self, store: Store, junk_type: str, level: float, seed: int = 0) -> Store:
        """Return a copy of `store` with junk units added (genuine units untouched)."""
        if level <= 0 or junk_type == "none":
            return store.with_units(store.units)
        rng = random.Random(f"{self.seed}|{seed}|{store.id}|{junk_type}|{level}")
        n = max(1, round(level * len(store.units)))
        types = JUNK_TYPES if junk_type == "mixed" else [junk_type]
        counts = {t: n // len(types) + (1 if i < n % len(types) else 0) for i, t in enumerate(types)}
        junk: List[MemoryUnit] = []
        for t, c in counts.items():
            junk += self._make(store, t, c, rng, start=len(junk))
        return store.with_units(store.units + junk)

    # ----------------------------------------------------------------- helpers
    def _sources(self, store: Store, n: int, rng: random.Random) -> List[MemoryUnit]:
        gold = [u for u in store.units if u.id in store.gold_ids()]
        out = []
        for _ in range(n):
            pool = gold if gold and rng.random() < self.targeted else store.units
            out.append(rng.choice(pool))
        return out

    def _make(self, store: Store, jtype: str, n: int, rng: random.Random, start: int) -> List[MemoryUnit]:
        if n <= 0:
            return []
        t_min = min(u.timestamp for u in store.units)
        t_max = max(u.timestamp for u in store.units)
        span = max(t_max - t_min, 1.0)
        out = []
        tries = 0
        srcs = self._sources(store, n * 3, rng)
        while len(out) < n and tries < n * 6:
            src = srcs[tries % len(srcs)]
            tries += 1
            text, ts, parent = None, None, None
            if jtype == "duplicate":
                text = _reword(src.text, rng)
                ts = src.timestamp + rng.uniform(0.01, 0.5) * span
                parent = src.id
            elif jtype in ("stale", "contradictory"):
                text = _mutate(src.text, rng, self.ent_vocab[store.id] or self.global_ents,
                               negate_ok=(jtype == "contradictory"))
                if text is None:
                    continue
                delta = rng.uniform(0.01, 0.5) * span
                ts = src.timestamp - delta if jtype == "stale" else src.timestamp + delta
                parent = src.id
            elif jtype == "irrelevant":
                others = [s for s in self.corpus.stores if s.id != store.id]
                if not others:
                    continue
                donor = rng.choice(rng.choice(others).units)
                text = donor.text
                ts = t_min + rng.random() * span
            elif jtype == "temporary":
                pool = [u for u in self.transient_pool if u.store_id != store.id]
                text = rng.choice(pool).text if pool else rng.choice(TRANSIENT_FALLBACK)
                ts = t_min + rng.random() * span
            else:
                raise KeyError(jtype)
            out.append(MemoryUnit(id=f"{store.id}/junk{start + len(out)}", text=text, timestamp=ts,
                                  store_id=store.id, junk_type=jtype, parent_id=parent))
        return out


def _cap_vocab(units) -> List[str]:
    v = set()
    for u in units:
        for m in CAP_SPAN_RE.finditer(u.text):
            w = m.group(0)
            if w.split()[0] not in NON_ENTITY_CAPS and len(w) > 2:
                v.add(w)
    return sorted(v)


def _reword(text: str, rng: random.Random) -> str:
    r = rng.random()
    if r < 0.3:
        return text
    if r < 0.6:
        head, sep, body = text.partition(": ")
        if sep and len(head) < 30:
            return f"{head}: {rng.choice(FILLERS)}{body[:1].lower()}{body[1:]}"
        return rng.choice(FILLERS) + text[:1].lower() + text[1:]
    if r < 0.85:
        words = text.split()
        if len(words) > 5:
            keep = [w for w in words if rng.random() > 0.12]
            return " ".join(keep or words)
        return text
    return re.sub(r"[!\.]+$", "", text).lower()


def _mutate(text: str, rng: random.Random, ent_vocab: Sequence[str], negate_ok: bool) -> Optional[str]:
    """Change one value-bearing slot: a number, year, month, identifier or a
    capitalised entity. Returns None when the text has nothing to change."""
    head, sep, body = text.partition(": ")
    if not (sep and len(head) < 40):
        head, sep, body = "", "", text
    options = []
    for m in NUMBER_RE.finditer(body):
        options.append(("num", m))
    for m in YEAR_RE.finditer(body):
        options.append(("year", m))
    for m in re.finditer(r"\b(" + "|".join(MONTHS) + r")\b", body):
        options.append(("month", m))
    for m in IDENT_RE.finditer(body):
        options.append(("ident", m))
    for m in CAP_SPAN_RE.finditer(body):
        before = body[:m.start()].rstrip()
        if m.group(0).split()[0] not in NON_ENTITY_CAPS and before and before[-1] not in ".!?\"":
            options.append(("ent", m))
    if negate_ok and re.search(AUX_NEG[0][0], body):
        options.append(("neg", None))
    if not options:
        return None
    kind, m = rng.choice(options)
    if kind == "neg":
        new_body = re.sub(AUX_NEG[0][0], AUX_NEG[0][1], body, count=1)
    else:
        old = m.group(0)
        if kind == "num":
            v = float(old.replace(",", ""))
            nv = v + rng.choice([-1, 1]) * max(1, round(abs(v) * rng.uniform(0.2, 0.8)))
            new = str(int(nv)) if "." not in old else f"{nv:.2f}"
        elif kind == "year":
            new = str(int(old) + rng.choice([-3, -2, -1, 1, 2]))
        elif kind == "month":
            new = rng.choice([x for x in MONTHS if x != old])
        elif kind == "ident":
            new = re.sub(r"\d", lambda d: str(rng.randint(0, 9)), old)
            if new == old:
                return None
        else:
            cands = [e for e in ent_vocab if e != old and e.lower() not in body.lower()]
            if not cands:
                return None
            new = rng.choice(cands)
        new_body = body[:m.start()] + new + body[m.end():]
    if new_body == body:
        return None
    return f"{head}{sep}{new_body}"
