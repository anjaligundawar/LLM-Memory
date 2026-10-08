"""
demo_pipeline.py  -- the slide-23 demo: VECTOR vs GRAPH, side by side,
on the polluted store (10 clean + 5 junk). Also saves results.csv and graph.png.

Command:   python demo_pipeline.py
"""
import csv
from toy_data import load_clean_units, TEST_QUERIES
from pollution_injector import inject_one_of_each
from vector_retriever import VectorRetriever
from graph_retriever import GraphRetriever
from metrics import precision_at_k, recall_at_k, reciprocal_rank

K = 3
LINE = "=" * 72


def flag(u):
    return f"   <-- {u.tags[0]}" if u.tags else ""


def main():
    clean_units = load_clean_units()
    units, ground_truth = inject_one_of_each(clean_units)
    print(f"[schema] {len(units)} memory units loaded "
          f"({len(clean_units)} clean, {len(ground_truth)} injected junk)")

    vector = VectorRetriever(units)
    graph = GraphRetriever(units, hops=2)
    rows = []

    for query, relevant in TEST_QUERIES.items():
        print(f"\n{LINE}\nQUERY: {query}    (correct: {relevant})\n{'-' * 72}")

        v_res, v_ms = vector.retrieve(query, k=K)
        print(f"[VECTOR] top-{K}  (latency {v_ms:.2f} ms)")
        for u, s in v_res:
            print(f"   score={s:.3f}  [{u.id}] {u.text[:52]}{flag(u)}")

        g_all, g_ms = graph.retrieve(query, k=None)
        g_res = g_all[:K]
        g_junk = [u.id for u, _ in g_all if u.is_junk()]
        print(f"[GRAPH]  {len(g_all)} hits via 2-hop traversal, {len(g_junk)} of them junk "
              f"{g_junk}  (latency {g_ms:.2f} ms)  -- first {K}:")
        for u, hop in g_res:
            print(f"   hop={hop}  [{u.id}] {u.text[:52]}{flag(u)}")

        for name, res, ms in [("vector", v_res, v_ms), ("graph", g_res, g_ms)]:
            ids = [u.id for u, _ in res]
            junk = sum(1 for u, _ in res if u.is_junk())
            p, r, rr = (precision_at_k(ids, relevant, K),
                        recall_at_k(ids, relevant, K), reciprocal_rank(ids, relevant))
            print(f"   {name:<6} P@{K}={p:.2f}  R@{K}={r:.2f}  MRR={rr:.2f}  junk in top-{K}={junk}/{K}")
            for metric, value in [(f"precision@{K}", p), (f"recall@{K}", r), ("mrr", rr),
                                  (f"junk@{K}", junk / K)]:
                rows.append({"condition": f"{name}-toy-5junk", "query": query,
                             "metric": metric, "value": round(value, 4),
                             "latency_ms": round(ms, 3), "seed": 0})

    n, e = graph.store.stats()
    print(f"\n{LINE}\nGraph nodes: {n}   Graph edges: {e}")

    with open("results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("Saved: results.csv")

    try:
        from draw_graph import draw
        draw(graph.store.G, "graph.png")
        print("Saved: graph.png")
    except ImportError:
        print("(install matplotlib to also save graph.png)")


if __name__ == "__main__":
    main()
