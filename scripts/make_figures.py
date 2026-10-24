"""Paper figures from the tables written by make_tables.py.

    python scripts/make_figures.py results/main
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"dense": "#1f77b4", "bm25": "#ff7f0e", "hybrid": "#2ca02c", "graph": "#d62728",
          "none": "#7f7f7f", "hard": "#d62728", "soft": "#1f77b4", "gated": "#2ca02c"}


def degradation(t: pd.DataFrame, fdir: Path, jtype: str = "mixed"):
    corpora = sorted(t.corpus.unique())
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.2 * len(corpora), 3), sharey=False, squeeze=False)
    for ax, c in zip(axes[0], corpora):
        for r, g in t[(t.corpus == c) & (t.jtype == jtype)].groupby("retriever"):
            g = g.sort_values("level")
            ax.plot(g.level, g["mean"], marker="o", label=r, color=COLORS.get(r))
            ax.fill_between(g.level, g.lo, g.hi, alpha=0.15, color=COLORS.get(r))
        ax.set_title(c)
        ax.set_xlabel("contamination level")
    axes[0][0].set_ylabel("Recall@5")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / f"degradation_{jtype}.pdf")
    plt.close(fig)


def harm_map(r: pd.DataFrame, fdir: Path):
    d = r[(r.component == "ALL") & (r.jtype == "mixed") & (r.level == 0.5)]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    for mode, marker in (("full", "o"), ("gated", "^")):
        x = d[d.hygiene == mode]
        ax.scatter(x.false_invalidation, x.junk_recall, marker=marker, label=mode, s=50)
        for _, row in x.iterrows():
            ax.annotate(row.corpus, (row.false_invalidation, row.junk_recall), fontsize=7,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("genuine memories wrongly flagged")
    ax.set_ylabel("junk memories caught")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / "harm_map.pdf")
    plt.close(fig)


def hygiene_bars(e: pd.DataFrame, fdir: Path, retriever: str = "dense", level: float = 0.5):
    d = e[(e.jtype == "mixed") & (e.retriever == retriever) & (e.level == level)]
    corpora = sorted(d.corpus.unique())
    modes = ["hard", "soft", "gated"]
    fig, ax = plt.subplots(figsize=(1.3 * len(corpora) + 1.5, 3))
    w = 0.8 / len(modes)
    for i, m in enumerate(modes):
        x = d[d.hygiene == m].set_index("corpus").reindex(corpora)
        pos = [j + i * w for j in range(len(corpora))]
        ax.bar(pos, x["mean"], w, color=COLORS[m], label=m,
               yerr=[x["mean"] - x.lo, x.hi - x["mean"]], capsize=2)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks([j + w for j in range(len(corpora))], corpora)
    ax.set_ylabel(f"Δ Recall@5 vs no hygiene")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fdir / f"hygiene_delta_{retriever}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    tdir, fdir = out / "tables", out / "figures"
    fdir.mkdir(exist_ok=True)
    degradation(pd.read_csv(tdir / "degradation.csv"), fdir)
    harm_map(pd.read_csv(tdir / "hygiene_rates.csv"), fdir)
    e = pd.read_csv(tdir / "hygiene_effect.csv")
    for r in e.retriever.unique():
        hygiene_bars(e, fdir, r)
    print("figures ->", fdir)
