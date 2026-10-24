"""Fit the gate's harm predictor from a finished grid (leave-one-corpus-out).

For every component, a ridge regression maps label-free store features plus
the component's flag rate to its false invalidation rate (share of genuine
memories it flags). For each corpus we fit on all OTHER corpora, so the gate
applied to a corpus has never seen it. Then re-run the grid with
gate.model_dir pointing at the output folder.

    python scripts/fit_gate.py results/main gate_models
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memhygiene.features import FEATURES  # noqa: E402
from memhygiene.hygiene.gate import GateModel  # noqa: E402
from memhygiene.runner import load_results  # noqa: E402

COLS = FEATURES + ["flag_rate"]


def fit(df, alpha=1.0) -> GateModel:
    X_all = df[COLS].to_numpy(float)
    mean, std = X_all.mean(0), X_all.std(0) + 1e-9
    m = GateModel(mean=mean.tolist(), std=std.tolist())
    for c, g in df.groupby("component"):
        X = (g[COLS].to_numpy(float) - mean) / std
        y = (g.flagged_genuine / g.n_genuine.clip(lower=1)).to_numpy(float)
        b = y.mean()
        w = np.linalg.solve(X.T @ X + alpha * np.eye(X.shape[1]), X.T @ (y - b))
        m.coef[c], m.intercept[c] = w.tolist(), float(b)
    return m


if __name__ == "__main__":
    res, out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "gate_models")
    eps = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    out.mkdir(parents=True, exist_ok=True)
    h = load_results(res, "hygiene")
    h = h[(h.hygiene == "full") & (h.component != "ALL")]
    report = {}
    for held in sorted(h.corpus.unique()):
        model = fit(h[h.corpus != held])
        model.save(out / f"loco_{held}.json")
        test = h[h.corpus == held]
        errs, correct = [], []
        for _, row in test.iterrows():
            pred = model.predict({f: row[f] for f in FEATURES}, {row.component: row.flag_rate})[row.component]
            true = row.flagged_genuine / max(1, row.n_genuine)
            errs.append(abs(pred - true))
            correct.append((pred <= eps) == (true <= eps))
        report[held] = {"mae": float(np.mean(errs)), "gate_decision_accuracy": float(np.mean(correct)),
                        "n": len(test)}
        print(held, report[held])
    fit(h).save(out / "all.json")
    (out / "loco_report.json").write_text(json.dumps(report, indent=2))
