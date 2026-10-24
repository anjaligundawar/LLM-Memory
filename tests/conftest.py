import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memhygiene.schema import Corpus, MemoryUnit, Query, Store  # noqa: E402

FACTS = [
    "Priya moved to Bangalore in March 2023 to join Infosys as a data analyst.",
    "Rohan lives in Pune and works at an automotive startup called VoltRide.",
    "Rohan did the Kedarkantha trek last December with 12 friends.",
    "Priya is learning guitar on weekends from a teacher named Arjun.",
    "Priya adopted a cat named Momo in November 2023.",
    "VoltRide raised a seed round of 2 million dollars in 2023.",
    "Priya switched jobs and joined Flipkart in January 2024.",
    "hey, how are you doing today?",
    "lol ok",
    "Rohan bought a blue Ather scooter in May 2023.",
]


def make_store(sid: str, offset: int = 0) -> Store:
    units = [MemoryUnit(id=f"{sid}/u{i}", text=t, timestamp=float(1000 * (i + offset)), store_id=sid)
             for i, t in enumerate(FACTS)]
    queries = [
        Query(id=f"{sid}/q0", store_id=sid, text="Where does Priya work now?", gold_ids=[f"{sid}/u6"],
              answer="Flipkart"),
        Query(id=f"{sid}/q1", store_id=sid, text="What is the name of Priya's cat?", gold_ids=[f"{sid}/u4"],
              answer="Momo"),
    ]
    return Store(id=sid, units=units, queries=queries)


@pytest.fixture
def corpus():
    return Corpus(name="toy", stores=[make_store("toy/a"), make_store("toy/b", 3)])


@pytest.fixture
def lme_file(tmp_path):
    inst = {
        "question_id": "q1", "question_type": "knowledge-update", "question": "Where do I work?",
        "answer": "Flipkart", "question_date": "2024/02/01 (Thu) 10:00",
        "haystack_session_ids": ["s1", "s2"], "haystack_dates": ["2023/05/20 (Sat) 02:21", "2024/01/15 (Mon) 09:00"],
        "haystack_sessions": [
            [{"role": "user", "content": "I work at Infosys.", "has_answer": True},
             {"role": "assistant", "content": "Nice!"}],
            [{"role": "user", "content": "I just joined Flipkart.", "has_answer": True},
             {"role": "assistant", "content": "Congrats!"}]],
        "answer_session_ids": ["s1", "s2"],
    }
    abs_inst = dict(inst, question_id="q2_abs")
    p = tmp_path / "lme.json"
    p.write_text(json.dumps([inst, abs_inst]))
    return p
