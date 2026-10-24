"""Shared data model.

A Corpus holds many Stores. A Store is one agent's long-term memory (one
conversation, one user, one story, one log window). Each Store has MemoryUnits
and the Queries asked against it. Gold evidence is always a list of unit ids
inside the same store.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional


@dataclass
class MemoryUnit:
    id: str
    text: str
    timestamp: float                  # seconds since epoch (or any monotone clock)
    store_id: str
    junk_type: Optional[str] = None   # None for genuine memories
    parent_id: Optional[str] = None   # genuine unit a junk unit was derived from
    meta: Dict = field(default_factory=dict)

    @property
    def is_junk(self) -> bool:
        return self.junk_type is not None


@dataclass
class Query:
    id: str
    store_id: str
    text: str
    gold_ids: List[str]
    answer: Optional[str] = None
    qtype: Optional[str] = None
    timestamp: Optional[float] = None


@dataclass
class Store:
    id: str
    units: List[MemoryUnit]
    queries: List[Query]
    meta: Dict = field(default_factory=dict)

    def unit_map(self) -> Dict[str, MemoryUnit]:
        return {u.id: u for u in self.units}

    def gold_ids(self) -> set:
        return {g for q in self.queries for g in q.gold_ids}

    def with_units(self, units: List[MemoryUnit]) -> "Store":
        return replace(self, units=list(units))


@dataclass
class Corpus:
    name: str
    stores: List[Store]
    meta: Dict = field(default_factory=dict)

    def n_units(self) -> int:
        return sum(len(s.units) for s in self.stores)

    def n_queries(self) -> int:
        return sum(len(s.queries) for s in self.stores)

    def summary(self) -> str:
        return (f"{self.name}: {len(self.stores)} stores, {self.n_units()} units, "
                f"{self.n_queries()} queries")
