"""
entity_extractor.py  -- MODULE 2 (Prachita): find entities in a sentence.

Review-2 heuristic: every Capitalised word is an entity
("Priya", "Infosys", "Bangalore", "March" ...). Possessive 's is removed.
Question words and sentence starters are ignored.

Final review: replace extract_entities() with spaCy NER or an LLM so we get
typed entities (Person / Organisation / Place / Date).
"""
import re

IGNORE = {
    "Where", "What", "When", "Who", "Why", "How", "Which", "Does", "Did",
    "Is", "The", "A", "An", "I", "She", "He", "They", "It", "Ugh", "Feeling",
    "New",  # "New Year" -> keep "Year" out too, below
    "Year",
}


def extract_entities(text):
    words = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    entities = []
    for w in words:
        w = re.sub(r"'s$", "", w)
        if w[0].isupper() and w not in IGNORE and w not in entities:
            entities.append(w)
    return entities
