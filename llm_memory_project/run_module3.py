"""
run_module3.py  -- run this file. It does the whole Module 3 demo:

  1. load 10 clean memories
  2. inject 5 junk memories (1 of each type)
  3. retrieve with the DIRTY store
  4. run the cleaner (hygiene v0)
  5. retrieve with the CLEANED store
  6. print junk fraction in top-3 before vs after
  7. bonus: show the injector at 0/10/25/50/75% (the grid for next review)

Command:   python run_module3.py
"""
from toy_data import load_clean_units, TEST_QUERIES
from pollution_injector import inject_one_of_each, noise_injector, CATEGORIES
from vector_retriever import VectorRetriever
from memory_cleaner import clean

K = 3
LINE = "=" * 72


def show(title, units, query):
    results, ms = VectorRetriever(units).retrieve(query, k=K)
    print(f"[{title}]  top-{K}  (latency {ms:.2f} ms)")
    for u, s in results:
        flag = f"   <-- {u.tags[0]}" if u.tags else ""
        print(f"   score={s:.3f}  [{u.id}] {u.text[:55]}{flag}")
    junk = sum(1 for u, _ in results if u.is_junk())
    return junk


def main():
    clean_units = load_clean_units()
    polluted, ground_truth = inject_one_of_each(clean_units)
    print(f"[schema] {len(polluted)} memory units loaded "
          f"({len(clean_units)} clean, {len(ground_truth)} injected junk)")
    print(f"[inject] ground truth: {ground_truth}\n")

    cleaned, removed, log = clean(polluted)
    print("\n".join(log))

    query = "Where does Priya work now?"
    print(f"\n{LINE}\nQUERY: {query}   (correct answer: {TEST_QUERIES[query]})\n{'-'*72}")
    before = show("VECTOR / DIRTY", polluted, query)
    after = show("VECTOR / CLEANED", cleaned, query)

    print(f"{LINE}")
    print(f"[result] junk fraction in top-{K} BEFORE cleaning: {before}/{K}")
    print(f"[result] junk fraction in top-{K} AFTER  v0 cleaning: {after}/{K}")

    print(f"\n{LINE}\nContamination grid preview (how many junk units the injector adds)")
    print("category        " + "  ".join(f"{int(p*100):>3}%" for p in [0, .1, .25, .5, .75]))
    for cat in CATEGORIES + ["mixed"]:
        row = [len(noise_injector(clean_units, cat, p)[1]) for p in [0, .1, .25, .5, .75]]
        print(f"{cat:<15} " + "  ".join(f"{n:>4}" for n in row))


if __name__ == "__main__":
    main()
