#!/usr/bin/env python3
"""Extract Kaggle IEEE-CIS zip and build ieee_cis_full.csv for ARL benchmarks."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

TRAIN_FILES = ("train_transaction.csv", "train_identity.csv")


def ingest_ieee_zip(
    zip_path: Path,
    *,
    raw_dir: Path | None = None,
    rebuild: bool = True,
) -> dict[str, str | int]:
    if not zip_path.exists():
        raise FileNotFoundError(f"IEEE zip not found: {zip_path}")

    target_raw = raw_dir or (ROOT / "data" / "fraud" / "raw")
    target_raw.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        missing = [name for name in TRAIN_FILES if name not in names]
        if missing:
            raise ValueError(f"Zip missing expected train files: {missing}. Found: {sorted(names)}")
        for name in TRAIN_FILES:
            archive.extract(name, path=target_raw)

    result: dict[str, str | int] = {
        "zip": str(zip_path),
        "train_transaction": str(target_raw / "train_transaction.csv"),
        "train_identity": str(target_raw / "train_identity.csv"),
    }

    if rebuild:
        from export_bundled_fraud_data import export_ieee_full_from_raw

        full_path = export_ieee_full_from_raw()
        if full_path is None:
            raise RuntimeError("export_ieee_full_from_raw failed after extraction")
        import pandas as pd

        row_count = len(pd.read_csv(full_path, usecols=["label"]))
        result["ieee_cis_full"] = str(full_path)
        result["rows"] = row_count

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kaggle IEEE-CIS fraud detection zip.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path.home() / "Downloads" / "ieee-fraud-detection.zip",
        help="Path to ieee-fraud-detection.zip",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Extract destination (default: data/fraud/raw)",
    )
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Only extract raw CSVs; do not rebuild ieee_cis_full.csv",
    )
    args = parser.parse_args()

    payload = ingest_ieee_zip(args.zip, raw_dir=args.raw_dir, rebuild=not args.no_rebuild)
    print("IEEE-CIS ingest complete:")
    for key, value in payload.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
