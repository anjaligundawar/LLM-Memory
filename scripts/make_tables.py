"""Build all result tables (CSV + LaTeX) from a finished grid.

    python scripts/make_tables.py results/main
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memhygiene import analysis as A  # noqa: E402
from memhygiene.runner import load_results  # noqa: E402

if __name__ == "__main__":
    out = Path(sys.argv[1])
    metric = sys.argv[2] if len(sys.argv) > 2 else "recall@5"
    tdir = out / "tables"
    tdir.mkdir(exist_ok=True)
    q, h = load_results(out), load_results(out, "hygiene")
    tables = {
        "degradation": A.degradation_table(q, metric),
        "pollution_effect": A.pollution_effect(q, metric),
        "vector_vs_graph": A.vector_vs_graph(q, metric),
        "hygiene_effect": A.hygiene_effect(q, metric),
        "hygiene_rates": A.hygiene_rates(h),
    }
    for name, t in tables.items():
        t.to_csv(tdir / f"{name}.csv", index=False)
        print(f"{name}: {len(t)} rows -> {tdir / (name + '.csv')}")

    # headline tables for the paper
    r = tables["hygiene_rates"]
    head = r[(r.component == "ALL") & (r.jtype == "mixed")].pivot_table(
        index=["corpus", "level"], columns="hygiene", values=["false_invalidation", "junk_recall"])
    head.columns = [f"{a}:{b}" for a, b in head.columns]
    A.to_latex(head.reset_index(), tdir / "hygiene_rates_mixed.tex")
    clean = r[(r.component != "ALL") & (r.level == 0) & (r.hygiene == "full")].pivot_table(
        index="corpus", columns="component", values="false_invalidation")
    A.to_latex(clean.reset_index(), tdir / "false_invalidation_clean.tex")
    e = tables["hygiene_effect"]
    e = e[(e.jtype == "mixed")].pivot_table(index=["corpus", "retriever", "level"], columns="hygiene",
                                            values="mean")
    A.to_latex(e.reset_index(), tdir / "hygiene_delta_mixed.tex")
    print("LaTeX tables written to", tdir)
