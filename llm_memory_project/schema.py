"""
schema.py  -- the shared MemoryUnit (owned by the whole team).

Every module reads/writes the SAME object:
  - Module 1 (Anjali)  fills  .embedding
  - Module 3 (Aksh)    fills  .tags   (junk-type labels)
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class MemoryUnit:
    id: str                 # primary key, e.g. "m01"
    text: str               # the memory sentence
    session: int            # which chat session it came from
    timestamp: str          # ISO date "YYYY-MM-DD"
    tags: List[str] = field(default_factory=list)   # e.g. ["stale"]
    embedding: Optional[list] = None                # filled by Module 1

    def is_junk(self) -> bool:
        return len(self.tags) > 0
