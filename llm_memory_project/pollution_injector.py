"""
pollution_injector.py  -- MODULE 3, PART 1 (Aksh)

Adds junk memories on purpose, at a controlled percentage, and records
exactly which ones are junk (ground_truth) so Module 4 can score them.

5 junk types:
  irrelevant     - unrelated memory from some other conversation
  duplicate      - same fact inserted again
  stale          - old fact that is no longer true
  contradictory  - opposite claim about the same person/attribute
  temporary      - small talk that should never be stored long-term
"""
import random
from schema import MemoryUnit

CATEGORIES = ["irrelevant", "duplicate", "stale", "contradictory", "temporary"]

# Hand-written junk for the toy corpus. For LoCoMo (next review) these are
# generated from the real conversations instead -- see README.
JUNK_POOL = {
    "irrelevant": [
        ("Arjun opened a small bakery in Chennai last month.", "2023-09-01"),
        ("The Chennai metro extended its line to the airport.", "2023-07-11"),
    ],
    "stale": [
        ("Priya works as a data analyst at Infosys in Bangalore.", "2023-04-01"),
        ("Rohan is still looking for investors for VoltRide.", "2023-06-01"),
    ],
    "contradictory": [
        ("Priya said she actually still works at Infosys.", "2024-02-10"),
        ("Rohan said he never went to Bangalore in 2023.", "2024-02-11"),
    ],
    "temporary": [
        ("Feeling sleepy today, will chat later.", "2023-08-14"),
        ("Ugh, it is raining heavily right now.", "2023-11-01"),
    ],
}

SHORT = {"irrelevant": "irr", "duplicate": "dup", "stale": "stale",
         "contradictory": "contra", "temporary": "temp"}


def generate(category, n, clean_units, start_id, rng):
    """Create n junk units of one category, each tagged with its type."""
    junk = []
    for i in range(n):
        new_id = f"m{start_id + i:02d}_{SHORT[category]}"
        if category == "duplicate":
            src = rng.choice(clean_units)           # copy a real memory word-for-word
            text, ts, session = src.text, src.timestamp, src.session + 1
        else:
            text, ts = JUNK_POOL[category][i % len(JUNK_POOL[category])]
            session = 9
        junk.append(MemoryUnit(new_id, text, session, ts, tags=[category]))
    return junk


def noise_injector(units, category, pct, seed=0):
    """
    units    : clean MemoryUnit list
    category : one of CATEGORIES, or "mixed" (split evenly across all 5)
    pct      : 0.0 - 0.75  contamination level
    returns  : (polluted_units, ground_truth {junk_id: category})
    """
    rng = random.Random(seed)
    n = round(len(units) * pct)
    cats = CATEGORIES if category == "mixed" else [category]

    junk, next_id = [], len(units) + 1
    for j, cat in enumerate(cats):
        share = n // len(cats) + (1 if j < n % len(cats) else 0)
        new = generate(cat, share, units, next_id, rng)
        junk += new
        next_id += len(new)

    polluted = units + junk
    ground_truth = {u.id: u.tags[0] for u in junk}
    return polluted, ground_truth


def inject_one_of_each(units):
    """The exact Review-2 setup: 10 clean + 1 of each type = 15 units."""
    rng = random.Random(0)
    m01 = next(u for u in units if u.id == "m01")
    junk = [
        MemoryUnit("m11_dup", m01.text, 2, "2023-05-02", tags=["duplicate"]),
        MemoryUnit("m12_stale", JUNK_POOL["stale"][0][0], 2, JUNK_POOL["stale"][0][1], tags=["stale"]),
        MemoryUnit("m13_contra", JUNK_POOL["contradictory"][0][0], 5, JUNK_POOL["contradictory"][0][1], tags=["contradictory"]),
        MemoryUnit("m14_irr", JUNK_POOL["irrelevant"][0][0], 9, JUNK_POOL["irrelevant"][0][1], tags=["irrelevant"]),
        MemoryUnit("m15_temp", JUNK_POOL["temporary"][0][0], 3, JUNK_POOL["temporary"][0][1], tags=["temporary"]),
    ]
    polluted = units + junk
    return polluted, {u.id: u.tags[0] for u in junk}
