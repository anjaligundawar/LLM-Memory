"""End-to-end QA: an LLM answers from the retrieved memories, and an LLM
judge grades the answer against the gold answer (LongMemEval-style)."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

from pydantic import BaseModel

from .schema import MemoryUnit

READER_SYSTEM = (
    "You answer questions about a user's past conversations using only the memories provided. "
    "Memories can be outdated, duplicated or contradictory; when facts conflict, prefer the most "
    "recent reliable one. Reply with a short answer only. If the memories do not contain the answer, "
    "reply exactly: I don't know.")

JUDGE_SYSTEM = (
    "You grade answers. Given a question, the gold answer and a model answer, decide whether the "
    "model answer is correct: it must contain or clearly imply the gold answer, and must not state a "
    "different value for it. Paraphrases and extra correct detail are fine.")


class Verdict(BaseModel):
    correct: bool


def _date(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except (OverflowError, OSError, ValueError):
        return f"t={ts:.0f}"


def answer(llm, question: str, memories: Sequence[MemoryUnit], asked_at: float | None = None) -> str:
    mem = "\n".join(f"[{_date(m.timestamp)}] {m.text}" for m in sorted(memories, key=lambda m: m.timestamp))
    when = f"\nThe question is asked on {_date(asked_at)}." if asked_at else ""
    return llm.text(READER_SYSTEM, f"Memories:\n{mem}\n{when}\nQuestion: {question}")


def judge(llm, question: str, gold: str, pred: str) -> bool:
    if not pred:
        return False
    v = llm.structured(JUDGE_SYSTEM, f"Question: {question}\nGold answer: {gold}\nModel answer: {pred}", Verdict)
    return bool(v.correct)
