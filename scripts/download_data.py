"""Download the public datasets into data/raw/.

    python scripts/download_data.py                  # all corpora
    python scripts/download_data.py locomo loghub    # a subset
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memhygiene.data import CORPORA  # noqa: E402
from memhygiene.data.download import DOWNLOADERS  # noqa: E402

if __name__ == "__main__":
    names = sys.argv[1:] or CORPORA
    for n in names:
        print(f"[{n}]")
        DOWNLOADERS[n]()
    print("done")
