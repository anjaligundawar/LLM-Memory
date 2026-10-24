"""Turn raw grid results into the paper's tables and statistics.

Every comparison is paired at the query level (same query, same seed, same
polluted store) and its CI comes from a store-level cluster bootstrap.
Seeds are averaged per query before bootstrapping.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from .eval.bootstrap import cluster_bootstrap_diff, cluster_bootstrap_mean
from .eval.metrics import audc

KEY = ["corpus", "store", "query"]


def _seed_mean(df: pd.DataFrame, by: List[str], metric: str) -> pd.DataFrame:
    return df.groupby(by + KEY, as_index=False)[metric].mean()


def degradation_table(q: pd.DataFrame, metric: str = "recall@5", hygiene: str = "none") -> pd.DataFrame:
    """Mean metric (with 95% CI) per corpus x retriever x junk type x level.
    Level 0 (clean store) is repeated for every junk type."""
    d = q[q.hygiene == hygiene]
    clean = d[d.level == 0]
    rows = []
    jtypes = sorted(set(d.jtype) - {"none"})
    for (c, r), g in d.groupby(["corpus", "retriever"]):
        for jt in jtypes:
            part = pd.concat([g[g.jtype == jt], clean[(clean.corpus == c) & (clean.retriever == r)]])
            for lvl, h in part.groupby("level"):
                m = h.groupby(KEY, as_index=False)[metric].mean()
                ci = cluster_bootstrap_mean(m[metric].values, m.store.values)
                rows.append(dict(corpus=c, retriever=r, jtype=jt, level=lvl, **ci, n=len(m)))
    t = pd.DataFrame(rows)
    au = (t.groupby(["corpus", "retriever", "jtype"])
           .apply(lambda g: audc(list(g.level), list(g["mean"])), include_groups=False)
           .rename("audc").reset_index())
    return t.merge(au, on=["corpus", "retriever", "jtype"])


def pollution_effect(q: pd.DataFrame, metric: str = "recall@5", hygiene: str = "none") -> pd.DataFrame:
    """Does pollution hurt? metric(level L) - metric(clean), paired per query."""
    d = q[q.hygiene == hygiene]
    clean = _seed_mean(d[d.level == 0], ["retriever"], metric).rename(columns={metric: "clean"})
    rows = []
    for (c, r, jt, lvl), g in d[d.level > 0].groupby(["corpus", "retriever", "jtype", "level"]):
        m = _seed_mean(g, [], metric).merge(clean[clean.retriever == r], on=KEY)
        ci = cluster_bootstrap_diff(m[metric].values, m["clean"].values, m.store.values)
        rows.append(dict(corpus=c, retriever=r, jtype=jt, level=lvl, **ci, n=len(m)))
    return pd.DataFrame(rows)


def vector_vs_graph(q: pd.DataFrame, metric: str = "recall@5", a: str = "dense", b: str = "graph") -> pd.DataFrame:
    """H1: does `a` degrade faster than `b`? Compares per-query drops
    (polluted - clean); a negative diff means `a` lost more."""
    d = q[q.hygiene == "none"]
    clean = _seed_mean(d[d.level == 0], ["retriever"], metric).rename(columns={metric: "clean"})
    rows = []
    for (c, jt, lvl), g in d[d.level > 0].groupby(["corpus", "jtype", "level"]):
        m = _seed_mean(g, ["retriever"], metric).merge(clean, on=KEY + ["retriever"])
        m["drop"] = m[metric] - m["clean"]
        w = m.pivot_table(index=KEY, columns="retriever", values="drop").dropna(subset=[a, b]).reset_index()
        if w.empty:
            continue
        ci = cluster_bootstrap_diff(w[a].values, w[b].values, w.store.values)
        rows.append(dict(corpus=c, jtype=jt, level=lvl, comparison=f"drop({a}) - drop({b})", **ci, n=len(w)))
    return pd.DataFrame(rows)


def hygiene_effect(q: pd.DataFrame, metric: str = "recall@5",
                   modes: Sequence[str] = ("hard", "soft", "gated")) -> pd.DataFrame:
    """Hygiene mode vs no hygiene on the same polluted store, paired per query."""
    rows = []
    for (c, r, jt, lvl), g in q.groupby(["corpus", "retriever", "jtype", "level"]):
        w = g.groupby(KEY + ["hygiene"], as_index=False)[[metric, f"junk@{metric.split('@')[1]}"]].mean()
        base = w[w.hygiene == "none"].set_index(KEY)
        for mode in modes:
            x = w[w.hygiene == mode].set_index(KEY).join(base, rsuffix="_none", how="inner").reset_index()
            if x.empty:
                continue
            ci = cluster_bootstrap_diff(x[metric].values, x[f"{metric}_none"].values, x.store.values)
            jk = f"junk@{metric.split('@')[1]}"
            junk_none, junk_mode = x[f"{jk}_none"].mean(), x[jk].mean()
            rows.append(dict(corpus=c, retriever=r, jtype=jt, level=lvl, hygiene=mode, **ci, n=len(x),
                             junk_none=junk_none, junk_mode=junk_mode,
                             junk_reduction=(junk_none / junk_mode) if junk_mode > 0 else np.inf))
    return pd.DataFrame(rows)


def hygiene_rates(h: pd.DataFrame) -> pd.DataFrame:
    """False invalidation rate (share of genuine memories flagged) and junk
    removal recall, per corpus x hygiene x component x junk type x level."""
    g = (h.groupby(["corpus", "hygiene", "component", "jtype", "level"])
          [["flagged_genuine", "n_genuine", "flagged_junk", "n_junk"]].sum().reset_index())
    g["false_invalidation"] = g.flagged_genuine / g.n_genuine.clip(lower=1)
    g["junk_recall"] = np.where(g.n_junk > 0, g.flagged_junk / g.n_junk.clip(lower=1), np.nan)
    flagged = g.flagged_genuine + g.flagged_junk
    g["flag_precision"] = np.where(flagged > 0, g.flagged_junk / flagged.clip(lower=1), np.nan)
    return g


def to_latex(df: pd.DataFrame, path, float_fmt: str = "%.3f", **kw) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format=float_fmt, escape=True, **kw))
