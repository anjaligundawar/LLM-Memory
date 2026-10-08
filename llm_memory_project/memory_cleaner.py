"""
memory_cleaner.py  -- MODULE 3, PART 2 (Aksh): Hygiene v0

Pass 1  DEDUP     : if two memories have cosine similarity >= 0.60,
                    drop the one with the later timestamp.
Pass 2  STALENESS : group memories by topic (e.g. "Priya's job"),
                    keep only the most recent TRUSTED one, drop the rest.
                    Contradictory units are excluded from the "most recent"
                    comparison (this was the bug fix shown on slide 17).

NOT done yet (next review): a real contradiction detector.
  v0 only knows a unit is contradictory because the injector tagged it.
  A real system has no tags, so v1 must DETECT contradictions itself.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DUP_THRESHOLD = 0.60

# Toy topic rules: (subject = first word of the memory, keywords that mean
# "this memory is about the subject's job"). v1: replace with LLM/NER.
TOPIC_RULES = {
    "priya_job_status": ("Priya", {"infosys", "flipkart", "works", "job", "jobs", "analyst"}),
    "rohan_job_status": ("Rohan", {"voltride", "startup", "investors", "works", "job"}),
}


def find_duplicates(units, log):
    vec = TfidfVectorizer(stop_words="english")
    sims = cosine_similarity(vec.fit_transform([u.text for u in units]))
    drop = set()
    log.append(f"[hygiene] duplicate-detection pairs (cosine >= {DUP_THRESHOLD}):")
    for i in range(len(units)):
        for j in range(i + 1, len(units)):
            if sims[i, j] >= DUP_THRESHOLD:
                a, b = units[i], units[j]
                later = a if a.timestamp > b.timestamp else b
                drop.add(later.id)
                log.append(f"    {a.id} <-> {b.id}   sim={sims[i, j]:.2f}")
    log.append(f"[hygiene] flagged as DUPLICATE, will be dropped: {sorted(drop)}")
    return drop


def topic_of(unit):
    words = unit.text.replace(",", " ").replace(".", " ").lower().split()
    first = unit.text.split()[0]
    for topic, (subject, keywords) in TOPIC_RULES.items():
        if first == subject and keywords & set(words):
            return topic
    return None


def is_contradictory(unit):
    # v0 CHEAT: uses the injector's label. Replace with a real detector in v1.
    return "contradictory" in unit.tags


def find_stale(units, log):
    buckets = {}
    for u in units:
        t = topic_of(u)
        if t:
            buckets.setdefault(t, []).append(u)

    drop = set()
    for topic, members in buckets.items():
        log.append(f"[hygiene] topic bucket '{topic}' has {len(members)} members: "
                   f"{[m.id for m in members]}")
        trusted = [m for m in members if not is_contradictory(m)]
        if len(trusted) <= 1:
            continue
        newest = max(trusted, key=lambda m: m.timestamp)
        stale = [m.id for m in trusted if m.id != newest.id]
        drop.update(stale)
        log.append(f"[hygiene]   keep newest trusted = {newest.id}; "
                   f"flagged as STALE: {stale}")
    return drop


def clean(units):
    """Returns (cleaned_units, removed_ids, log_lines)."""
    log = []
    dup = find_duplicates(units, log)
    remaining = [u for u in units if u.id not in dup]
    stale = find_stale(remaining, log)
    removed = dup | stale
    cleaned = [u for u in units if u.id not in removed]
    log.append(f"[hygiene] {len(units)} units -> {len(cleaned)} after v0 cleaning "
               f"({len(removed)} removed: {sorted(removed)})")
    left_contra = [u.id for u in cleaned if is_contradictory(u)]
    if left_contra:
        log.append(f"[hygiene] {left_contra} intentionally NOT removed -- "
                   f"contradiction detection is next-review work")
    return cleaned, removed, log
