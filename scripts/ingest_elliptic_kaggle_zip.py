#!/usr/bin/env python3
"""Extract Kaggle Elliptic Bitcoin zip and build elliptic_fraud.csv for ARL benchmarks."""

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

ELLIPTIC_FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_classes.csv",
    "elliptic_txs_edgelist.csv",
)


def ingest_elliptic_zip(
    zip_path: Path,
    *,
    raw_dir: Path | None = None,
    rebuild: bool = True,
    max_rows: int | None = None,
) -> dict[str, str | int]:
    if not zip_path.exists():
        raise FileNotFoundError(f"Elliptic zip not found: {zip_path}")

    target_raw = raw_dir or (ROOT / "data" / "fraud" / "raw" / "elliptic")
    target_raw.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for name in ELLIPTIC_FILES:
            matches = [item for item in names if item.endswith(name)]
            if not matches:
                if name == "elliptic_txs_edgelist.csv":
                    continue
                raise ValueError(f"Zip missing {name}. Found: {sorted(names)[:20]}...")
            archive.extract(matches[0], path=target_raw)
            extracted = target_raw / matches[0]
            if extracted.name != name:
                extracted.rename(target_raw / name)

    result: dict[str, str | int] = {
        "zip": str(zip_path),
        "raw_dir": str(target_raw),
    }
    if rebuild:
        from export_elliptic_baf_fraud_data import export_elliptic_from_raw

        output = export_elliptic_from_raw(raw_dir=target_raw, max_rows=max_rows)
        if output is None:
            raise RuntimeError("export_elliptic_from_raw failed after extraction")
        import pandas as pd

        result["elliptic_fraud"] = str(output)
        result["rows"] = len(pd.read_csv(output, usecols=["label"]))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Kaggle Elliptic Bitcoin dataset zip.")
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path.home() / "Downloads" / "elliptic-data-set.zip",
        help="Path to elliptic-data-set.zip",
    )
    parser.add_argument("--raw-dir", type=Path, default=None)
    parser.add_argument("--no-rebuild", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for dev runs")
    args = parser.parse_args()
    payload = ingest_elliptic_zip(
        args.zip,
        raw_dir=args.raw_dir,
        rebuild=not args.no_rebuild,
        max_rows=args.max_rows,
    )
    for key, value in payload.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
