"""
graph_retriever.py  -- MODULE 2 (Prachita): k-hop BFS retrieval.

1. find entities in the query           ("Priya")
2. BFS out from them, max 2 hops        (Priya -> Rohan -> Pune ...)
3. every edge walked gives its memory   (memory_id + text)
4. closer memories (1-hop) come first

Cost: O(V + E). 2 hops reaches "memories about related people", which
vector search cannot do.
"""
import time
from entity_extractor import extract_entities
from graph_store import GraphStore


class GraphRetriever:
    def __init__(self, units, hops=2):
        self.store = GraphStore(units)
        self.hops = hops

    def retrieve(self, query, k=3):
        """returns ([(MemoryUnit, hop_distance), ...] first k, latency_ms).
        k=None returns ALL hits."""
        t0 = time.perf_counter()
        G = self.store.G
        q_entities = [e for e in extract_entities(query) if e in G]

        # BFS: distance of every reachable node from the query entities
        dist = {e: 0 for e in q_entities}
        frontier = list(q_entities)
        for d in range(1, self.hops + 1):
            nxt = []
            for node in frontier:
                for nb in G.neighbors(node):
                    if nb not in dist:
                        dist[nb] = d
                        nxt.append(nb)
            frontier = nxt

        # collect memories on every edge leaving a node closer than `hops`
        hits = {}                                   # memory_id -> hop distance
        for node, d in dist.items():
            if d >= self.hops:
                continue
            for _, _, data in G.edges(node, data=True):
                mid = data["memory_id"]
                hop = d + 1
                if mid not in hits or hop < hits[mid]:
                    hits[mid] = hop

        # 1-hop first; among equals, memories that mention MORE query entities first
        def rank(mid):
            text_ents = set(extract_entities(self.store.units[mid].text))
            return (hits[mid], -len(text_ents & set(q_entities)), mid)

        ordered = sorted(hits, key=rank)
        result = [(self.store.units[m], hits[m]) for m in (ordered if k is None else ordered[:k])]
        latency_ms = (time.perf_counter() - t0) * 1000
        return result, latency_ms
