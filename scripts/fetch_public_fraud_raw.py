#!/usr/bin/env python3
"""Download real Elliptic (PyG mirror) and BAF Base (Hugging Face mirror) without Kaggle API."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.data_export.fetch_raw import fetch_baf_raw, fetch_elliptic_raw
from adaptive_reliability_layer.workspace import fraud_data_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--elliptic-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--baf-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--elliptic-only", action="store_true")
    parser.add_argument("--baf-only", action="store_true")
    args = parser.parse_args()

    fraud_root = fraud_data_dir(root=ROOT)
    elliptic_dir = args.elliptic_dir or (fraud_root / "raw" / "elliptic")
    baf_dir = args.baf_dir or (fraud_root / "raw" / "baf")

    if not args.baf_only:
        files = fetch_elliptic_raw(elliptic_dir)
        print("Elliptic raw files:")
        for path in files:
            print(f"  {path} ({path.stat().st_size // 1_000_000} MB)")
    if not args.elliptic_only:
        baf = fetch_baf_raw(baf_dir)
        print(f"BAF raw: {baf} ({baf.stat().st_size // 1_000_000} MB)")


if __name__ == "__main__":
    main()
