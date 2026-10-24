"""Run an experiment grid from a YAML config.

    python scripts/run_grid.py configs/smoke.yaml
    python scripts/run_grid.py configs/main.yaml --corpora locomo clutrr
"""
import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memhygiene.runner import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--corpora", nargs="*", help="only these corpora from the config")
    ap.add_argument("--out", help="override the output folder")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.corpora:
        cfg["corpora"] = {k: v for k, v in cfg["corpora"].items() if k in a.corpora}
    if a.out:
        cfg["out"] = a.out
    print(f"results -> {run(cfg)}")
