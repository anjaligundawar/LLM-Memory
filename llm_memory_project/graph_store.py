"""
graph_store.py  -- MODULE 2 (Prachita): build the entity graph.

  node = entity string      ("Priya", "Infosys", ...)
  edge = RELATED_TO         two entities mentioned in the SAME memory
         edge properties:   memory_id, text   (so every hit is traceable)

A MultiGraph is used so two different memories that mention the same pair
of entities give two separate edges (no memory is lost).

Final review: move the same nodes/edges into Neo4j (see README).
"""
import networkx as nx
from entity_extractor import extract_entities


class GraphStore:
    def __init__(self, units):
        self.units = {u.id: u for u in units}
        self.G = nx.MultiGraph()
        for u in units:
            ents = extract_entities(u.text)
            for e in ents:
                self.G.add_node(e)
                self.G.nodes[e].setdefault("memories", []).append(u.id)
            for i in range(len(ents)):
                for j in range(i + 1, len(ents)):
                    self.G.add_edge(ents[i], ents[j], relation="RELATED_TO",
                                    memory_id=u.id, text=u.text)

    def stats(self):
        return self.G.number_of_nodes(), self.G.number_of_edges()
