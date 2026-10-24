"""Small text utilities shared by pollution, hygiene and the graph retriever."""
from __future__ import annotations

import re
from typing import List

TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_'\-\.:/]*[A-Za-z0-9_]|[A-Za-z0-9]")
NUMBER_RE = re.compile(r"(?<![A-Za-z_\d\.,])\d+(?:[\.,]\d+)?(?![A-Za-z_\d])")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_RE = re.compile(r"\b(" + "|".join(MONTHS) + r")\b")
CAP_SPAN_RE = re.compile(r"\b[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*\b")
IDENT_RE = re.compile(r"\b(?=[A-Za-z0-9_\-\.:/]*\d)(?=[A-Za-z0-9_\-\.:/]*[A-Za-z_])[A-Za-z0-9_\-\.:/]{4,}\b"
                      r"|\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down
during each few for from further had hadn't has hasn't have haven't having he he'd he'll he's her here
here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it it's its
itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours
ourselves out over own same shan't she she'd she'll she's should shouldn't so some such than that
that's the their theirs them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what
what's when when's where where's which while who who's whom why why's with won't would wouldn't you
you'd you'll you're you've your yours yourself yourselves just really also so yeah oh ok okay well
gonna got get like thing things lot lots much many will one would could us
""".split())

# Words that start sentences or questions and are not entities.
NON_ENTITY_CAPS = set(MONTHS + WEEKDAYS + [
    "I", "The", "A", "An", "What", "When", "Where", "Who", "Why", "How", "Which", "Did", "Does",
    "Do", "Is", "Are", "Was", "Were", "Can", "Could", "Would", "Will", "Should", "Has", "Have",
    "Hey", "Hi", "Hello", "Thanks", "Thank", "Yes", "No", "Oh", "Wow", "That", "This", "It",
    "My", "Your", "We", "You", "He", "She", "They", "So", "But", "And", "Yeah", "Sure", "Great",
    "Awesome", "Nice", "Haha", "Ok", "Okay", "Well", "Also", "Just", "Not", "If", "In", "On", "At",
])

PHATIC = set("""hi hey hello thanks thank bye goodbye lol haha ok okay sure cool nice great awesome
wow yeah yep nope brb ttyl gtg night morning cheers welcome sounds good glad""".split())


def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def content_tokens(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


def entities(text: str) -> List[str]:
    """Cheap entity extractor: capitalised spans (minus sentence starters),
    identifiers with digits, years and month names. Used when spaCy is absent
    and for slot detection, where speed matters more than precision."""
    out: List[str] = []
    for m in CAP_SPAN_RE.finditer(text):
        words = [w for w in m.group(0).split() if w not in NON_ENTITY_CAPS]
        if words:
            out.append(" ".join(words))
    out += [m.group(0) for m in IDENT_RE.finditer(text)]
    out += [m.group(0) for m in YEAR_RE.finditer(text)]
    out += MONTH_RE.findall(text)
    seen, uniq = set(), []
    for e in out:
        k = e.lower()
        if k not in seen:
            seen.add(k)
            uniq.append(e)
    return uniq


def slots(text: str) -> set:
    """Value-bearing tokens: numbers, entity names, dates. Two memories about
    the same thing that differ only in slots are candidate updates/conflicts."""
    s = {m.group(0).lower() for m in NUMBER_RE.finditer(text)}
    s |= {e.lower() for e in entities(text)}
    return s


def informativeness(text: str) -> float:
    """0..1 score: share of content tokens that are not phatic, damped for very
    short texts. Low values mark small talk / transient state."""
    toks = tokenize(text)
    if not toks:
        return 0.0
    content = [t for t in toks if t not in STOPWORDS and t not in PHATIC and len(t) > 1]
    ratio = len(content) / len(toks)
    length_factor = min(1.0, len(content) / 6.0)
    return ratio * length_factor
