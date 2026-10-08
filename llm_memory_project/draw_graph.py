"""
draw_graph.py  -- MODULE 2 (Prachita): save a picture of the entity graph
(the slide-24 screenshot).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


def draw(G, path="graph.png"):
    simple = nx.Graph(G)                         # merge parallel edges for drawing
    pos = nx.spring_layout(simple, seed=7, k=0.9)
    plt.figure(figsize=(11, 8))
    nx.draw_networkx_edges(simple, pos, alpha=0.4)
    nx.draw_networkx_nodes(simple, pos, node_size=1600, node_color="#9ecae1")
    nx.draw_networkx_labels(simple, pos, font_size=10)
    plt.title(f"Entity graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()
