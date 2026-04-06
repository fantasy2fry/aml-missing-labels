"""
AML Project 1 – data_download.py
=================================
One-time script that fetches all four datasets from the UCI ML Repository
and saves them to ``data/raw/`` as CSV files.

Run once before anything else:

    python data_download.py

Output structure
----------------
    data/
    └── raw/
        ├── spambase_X.csv
        ├── spambase_y.csv
        ├── sonar_X.csv
        ├── sonar_y.csv
        ├── breast_cancer_X.csv
        ├── breast_cancer_y.csv
        ├── phishing_X.csv
        └── phishing_y.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from ucimlrepo import fetch_ucirepo

# ── Configuration ─────────────────────────────────────────────────────────────

# Always place data/raw/ next to the project root, regardless of where
# the script is called from. __file__ resolves to src/data_download.py,
# so .parent.parent gives the project root.
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

DATASETS: dict[str, int] = {
    "spambase":      94,
    "sonar":        151,
    "breast_cancer":  17,
    "phishing":     327,
}


# ── Download logic ────────────────────────────────────────────────────────────

def download_all(force: bool = False) -> None:
    """Download all datasets and save them to ``data/raw/``.

    Parameters
    ----------
    force : bool, default False
        If True, re-download even if the CSV files already exist.
        If False, skip datasets whose files are already present.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, uid in DATASETS.items():
        x_path = RAW_DIR / f"{name}_X.csv"
        y_path = RAW_DIR / f"{name}_y.csv"

        if not force and x_path.exists() and y_path.exists():
            print(f"[skip]     '{name}' already exists – use --force to re-download.")
            continue

        print(f"[download] Fetching '{name}' (UCI ID={uid})…", end=" ", flush=True)
        try:
            repo = fetch_ucirepo(id=uid)
            X: pd.DataFrame = repo.data.features
            y: pd.DataFrame = repo.data.targets

            X.to_csv(x_path, index=False)
            y.to_csv(y_path, index=False)
            print(f"done  (X={X.shape}, y={y.shape})")

        except Exception as exc:
            print(f"FAILED\n  → {exc}", file=sys.stderr)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    force = "--force" in sys.argv
    if force:
        print("Running in force mode – all datasets will be re-downloaded.\n")

    download_all(force=force)

    print("\nAll done. Files saved to:")
    for p in sorted(RAW_DIR.glob("*.csv")):
        print(f"  {p}  ({p.stat().st_size / 1024:.1f} KB)")