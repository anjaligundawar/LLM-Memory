"""Natural junk: memories written by an LLM memory writer, not injected.

A naive "ADD-only" writer (the default behaviour of many agent memory
pipelines) reads each conversation session and stores standalone facts.
Over many sessions it naturally produces duplicates, outdated facts,
contradictions and small talk. An LLM annotator then labels every stored
memory against its most similar neighbours, giving natural junk labels to
score hygiene against (validate a random sample by hand before reporting).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import List, Literal

import numpy as np
from pydantic import BaseModel

from ..schema import MemoryUnit, Store

WRITER_SYSTEM = (
    "You are the long-term memory writer of a personal assistant. From the conversation session below, "
    "write the facts worth remembering about the people in it, one standalone sentence each, with full "
    "names instead of pronouns. Write each fact as it is stated in this session, even if you think it "
    "was mentioned before.")

ANNOTATOR_SYSTEM = (
    "You audit an assistant's long-term memory. You see one TARGET memory and other memories from the "
    "same store, each with the date it was written. Label the TARGET:\n"
    "valid: a correct, useful fact that should be kept\n"
    "duplicate: states the same fact as an OLDER memory listed\n"
    "outdated: was true but a NEWER memory listed shows it changed\n"
    "contradictory: conflicts with another memory and is the less trustworthy one\n"
    "transient: small talk or a momentary state not worth keeping long-term")


class Facts(BaseModel):
    facts: List[str]


class Label(BaseModel):
    label: Literal["valid", "duplicate", "outdated", "contradictory", "transient"]


def _sessions(store: Store):
    by = defaultdict(list)
    for u in store.units:
        by[u.meta.get("session")].append(u)
    return sorted(by.values(), key=lambda us: us[0].timestamp)


def write_memories(llm, store: Store, max_sessions: int | None = None) -> List[MemoryUnit]:
    out: List[MemoryUnit] = []
    for si, turns in enumerate(_sessions(store)[:max_sessions]):
        date = datetime.fromtimestamp(turns[0].timestamp).strftime("%Y-%m-%d")
        dialogue = "\n".join(t.text for t in turns)
        facts = llm.structured(WRITER_SYSTEM, f"Session date: {date}\n\n{dialogue}", Facts).facts
        for fi, f in enumerate(facts):
            out.append(MemoryUnit(id=f"{store.id}/nat{si}_{fi}", text=f.strip(), timestamp=turns[0].timestamp + fi,
                                  store_id=store.id, meta={"session": turns[0].meta.get("session"),
                                                            "source_ids": [t.id for t in turns]}))
    return out


def annotate(llm, units: List[MemoryUnit], vectors: np.ndarray, neighbours: int = 6) -> List[str]:
    V = np.asarray(vectors, dtype=np.float32)
    S = V @ V.T
    np.fill_diagonal(S, -1)
    labels = []
    for i, u in enumerate(units):
        # never show the target as one of its own neighbours (happens when the
        # store has <= `neighbours` memories, since the diagonal only sorts last)
        nb = [j for j in np.argsort(-S[i]) if j != i][:neighbours]
        ctx = "\n".join(f"- [{datetime.fromtimestamp(units[j].timestamp):%Y-%m-%d}] {units[j].text}"
                        for j in sorted(nb, key=lambda j: units[j].timestamp))
        user = (f"TARGET [{datetime.fromtimestamp(u.timestamp):%Y-%m-%d}]: {u.text}\n\n"
                f"Other memories:\n{ctx}")
        labels.append(llm.structured(ANNOTATOR_SYSTEM, user, Label).label)
    return labels
