#!/usr/bin/env python3
"""Extract Kaggle BAF (Bank Account Fraud) zip and build baf_base_fraud.csv."""

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


def ingest_baf_zip(
    zip_path: Path,
    *,
    raw_dir: Path | None = None,
    variant: str = "Base",
    rebuild: bool = True,
    max_rows: int = 250_000,
) -> dict[str, str | int]:
    if not zip_path.exists():
        raise FileNotFoundError(f"BAF zip not found: {zip_path}")

    target_raw = raw_dir or (ROOT / "data" / "fraud" / "raw" / "baf")
    target_raw.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            archive.extract(name, path=target_raw)
            extracted = target_raw / name
            flat = target_raw / Path(name).name
            if extracted != flat:
                flat.parent.mkdir(parents=True, exist_ok=True)
                extracted.replace(flat)

    result: dict[str, str | int] = {"zip": str(zip_path), "raw_dir": str(target_raw), "variant": variant}
    if rebuild:
        from export_elliptic_baf_fraud_data import export_baf_from_raw

        output = export_baf_from_raw(raw_dir=target_raw, variant=variant, max_rows=max_rows)
        if output is None:
            raise RuntimeError("export_baf_from_raw failed — no CSV found in extracted zip")
        import pandas as pd

        result["baf_base_fraud"] = str(output)
        result["rows"] = len(pd.read_csv(output, usecols=["label"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kaggle Bank Account Fraud (BAF) dataset zip.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path.home() / "Downloads" / "bank-account-fraud-dataset-neurips-2022.zip",
        help="Path to BAF Kaggle zip",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--variant", default="Base", help="BAF variant name substring (default: Base)")
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--max-rows", type=int, default=250_000)
    args = parser.parse_args()
    payload = ingest_baf_zip(
        args.zip,
        raw_dir=args.raw_dir,
        variant=args.variant,
        rebuild=not args.no_rebuild,
        max_rows=args.max_rows,
    )
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
