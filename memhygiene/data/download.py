"""Fetch raw datasets from their official public sources.

Everything lands under data/raw/<corpus>/. Files already present are skipped,
so this is safe to re-run.

Sources
  locomo      github.com/snap-research/locomo           (Maharana et al., ACL 2024)
  longmemeval huggingface.co/datasets/xiaowu0162/longmemeval-cleaned  (Wu et al., ICLR 2025)
  mab         huggingface.co/datasets/ai-hyz/MemoryAgentBench          (Hu et al., 2025)
  clutrr      CSVs behind huggingface.co/datasets/CLUTRR/v1            (Sinha et al., EMNLP 2019)
  loghub      github.com/logpai/loghub, 2k-line labelled samples        (Zhu et al., ISSRE 2023)
"""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

RAW = Path(os.environ.get("MEMHYG_DATA", "data")) / "raw"

LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
CLUTRR_URL = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/{config}/{split}.csv"
CLUTRR_CONFIGS = ["gen_train234_test2to10", "rob_train_irr_23_test_all_23"]
LOGHUB_URL = "https://raw.githubusercontent.com/logpai/loghub/master/{system}/{system}_2k.log_structured.csv"
LOGHUB_SYSTEMS = ["HDFS", "Hadoop", "Spark", "Zookeeper", "OpenStack", "BGL", "HPC", "Thunderbird",
                  "Linux", "Mac", "Android", "HealthApp", "Apache", "OpenSSH", "Proxifier"]
LONGMEMEVAL_REPO = "xiaowu0162/longmemeval-cleaned"
LONGMEMEVAL_FILES = {"s": "longmemeval_s_cleaned.json", "oracle": "longmemeval_oracle.json",
                     "m": "longmemeval_m_cleaned.json"}
MAB_REPO = "ai-hyz/MemoryAgentBench"
MAB_FILE = "data/Conflict_Resolution-00000-of-00001.parquet"


def _fetch(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    return dest


def _hf(repo: str, filename: str, dest_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"  downloading hf://datasets/{repo}/{filename}")
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                                local_dir=str(dest_dir)))


def download_locomo() -> Path:
    return _fetch(LOCOMO_URL, RAW / "locomo" / "locomo10.json")


def download_clutrr() -> Path:
    for cfg in CLUTRR_CONFIGS:
        _fetch(CLUTRR_URL.format(config=cfg, split="test"), RAW / "clutrr" / cfg / "test.csv")
    return RAW / "clutrr"


def download_loghub() -> Path:
    for system in LOGHUB_SYSTEMS:
        try:
            _fetch(LOGHUB_URL.format(system=system), RAW / "loghub" / f"{system}_2k.log_structured.csv")
        except Exception as e:  # a few systems are occasionally renamed upstream
            print(f"  skipped {system}: {e}")
    return RAW / "loghub"


def download_longmemeval(variant: str = "s") -> Path:
    dest = RAW / "longmemeval" / LONGMEMEVAL_FILES[variant]
    if dest.exists():
        return dest
    return _hf(LONGMEMEVAL_REPO, LONGMEMEVAL_FILES[variant], RAW / "longmemeval")


def download_mab() -> Path:
    dest = RAW / "mab" / MAB_FILE
    if dest.exists():
        return dest
    return _hf(MAB_REPO, MAB_FILE, RAW / "mab")


DOWNLOADERS = {
    "locomo": download_locomo,
    "longmemeval": download_longmemeval,
    "mab_conflict": download_mab,
    "clutrr": download_clutrr,
    "loghub": download_loghub,
}
