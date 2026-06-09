#!/usr/bin/env python3
"""Download and preprocess the NASA CMAPSS turbofan degradation dataset.

Produces data/cmapss/cmapss_FD001.csv ... cmapss_FD004.csv in project-standard
format with columns:
  unit, time_cycle, op1, op2, op3, s1..s21, rul, label

If the download fails, a synthetic fallback is generated so the benchmark
runner can still be tested end-to-end.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd

# NASA public download URL (no auth required)
CMAPSS_NASA_URL = (
    "https://data.nasa.gov/download/mznv-hvai/application%2Fzip"
)

# Column names: unit, time_cycle, 3 operating settings, 21 sensor readings
_COLUMN_NAMES = (
    ["unit", "time_cycle", "op1", "op2", "op3"]
    + [f"s{i}" for i in range(1, 22)]
)
# The text files have 26 columns total; the last is always NaN (trailing space)
_ALL_COLUMNS = _COLUMN_NAMES + ["_trailing"]

DATASET_IDS = ["FD001", "FD002", "FD003", "FD004"]

# Binary-failure threshold: a unit is considered "failed" if RUL ≤ 30 cycles
FAILURE_THRESHOLD = 30


def _cmapss_dir() -> Path:
    path = ROOT / "data" / "cmapss"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _parse_cmapss_text(raw: bytes | str, dataset_id: str) -> pd.DataFrame:
    """Parse a CMAPSS whitespace-delimited text file into a DataFrame."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8")
    else:
        text = raw
    frame = pd.read_csv(
        io.StringIO(text),
        sep=r"\s+",
        header=None,
        names=_ALL_COLUMNS,
        engine="python",
    )
    # Drop the always-NaN trailing column
    frame = frame.drop(columns=["_trailing"], errors="ignore")
    # Keep only the expected 26 columns if somehow parsed differently
    frame = frame[[col for col in _COLUMN_NAMES if col in frame.columns]]
    frame = frame.astype(float)
    frame["unit"] = frame["unit"].astype(int)
    frame["time_cycle"] = frame["time_cycle"].astype(int)
    return frame


def _compute_rul(frame: pd.DataFrame) -> pd.DataFrame:
    """Add `rul` and binary `label` columns."""
    max_cycles = frame.groupby("unit")["time_cycle"].transform("max")
    frame = frame.copy()
    frame["rul"] = (max_cycles - frame["time_cycle"]).astype(int)
    frame["label"] = (frame["rul"] <= FAILURE_THRESHOLD).astype(int)
    return frame


def _sort_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["unit", "time_cycle"]).reset_index(drop=True)


def export_cmapss_from_zip(zip_bytes: bytes, *, output_dir: Path) -> dict[str, Path]:
    """Parse zip bytes and write one CSV per dataset.  Returns {id: path}."""
    outputs: dict[str, Path] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        for dataset_id in DATASET_IDS:
            # Files inside the zip may be flat or inside a sub-directory
            candidates = [
                n for n in names
                if f"train_{dataset_id}" in n and n.endswith(".txt")
            ]
            if not candidates:
                print(f"  WARNING: train_{dataset_id}.txt not found in zip; skipping.")
                continue
            raw = zf.read(candidates[0])
            frame = _parse_cmapss_text(raw, dataset_id)
            frame = _compute_rul(frame)
            frame = _sort_frame(frame)
            out_path = output_dir / f"cmapss_{dataset_id}.csv"
            frame.to_csv(out_path, index=False)
            outputs[dataset_id] = out_path
    return outputs


def export_cmapss_from_dir(raw_dir: Path, *, output_dir: Path) -> dict[str, Path]:
    """Parse already-downloaded .txt files in raw_dir."""
    outputs: dict[str, Path] = {}
    for dataset_id in DATASET_IDS:
        txt_path = raw_dir / f"train_{dataset_id}.txt"
        if not txt_path.exists():
            # Also look without subdirectory nesting
            alt = raw_dir / f"CMAPSSData" / f"train_{dataset_id}.txt"
            if alt.exists():
                txt_path = alt
            else:
                print(f"  WARNING: {txt_path} not found; skipping {dataset_id}.")
                continue
        raw = txt_path.read_bytes()
        frame = _parse_cmapss_text(raw, dataset_id)
        frame = _compute_rul(frame)
        frame = _sort_frame(frame)
        out_path = output_dir / f"cmapss_{dataset_id}.csv"
        frame.to_csv(out_path, index=False)
        outputs[dataset_id] = out_path
    return outputs


def export_cmapss_synthetic_fallback(
    *,
    output_dir: Path,
    n_units: int = 20,
    n_cycles: int = 200,
    seed: int = 42,
) -> dict[str, Path]:
    """Generate a synthetic dataset with monotonically drifting sensors.

    Uses 20 units so a per-unit train/test split gives the source model
    realistic training exposure to both healthy AND degraded cycles (i.e.
    positive examples appear in training).  This mirrors how real CMAPSS is
    used: different engine *units* are in train vs test, not different time
    windows of the same unit.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for unit in range(1, n_units + 1):
        unit_cycles = n_cycles + rng.integers(-20, 21)
        for t in range(1, unit_cycles + 1):
            progress = t / unit_cycles  # 0→1 as engine approaches failure
            op_settings = rng.normal(0.0, 0.1, size=3)
            sensors_up = (
                0.5 * (1.0 - progress)
                + rng.normal(0.0, 0.05, size=10)
                + 0.3 * progress * rng.normal(1.0, 0.1, size=10)
            )
            sensors_down = (
                0.5 * progress
                + rng.normal(0.0, 0.05, size=11)
                - 0.3 * progress * rng.normal(1.0, 0.1, size=11)
            )
            sensors = np.concatenate([sensors_up, sensors_down])
            rows.append(
                [unit, t]
                + list(op_settings)
                + list(sensors)
            )
    frame = pd.DataFrame(rows, columns=_COLUMN_NAMES)
    frame["unit"] = frame["unit"].astype(int)
    frame["time_cycle"] = frame["time_cycle"].astype(int)
    frame = _compute_rul(frame)
    frame = _sort_frame(frame)

    outputs: dict[str, Path] = {}
    for dataset_id in DATASET_IDS:
        out_path = output_dir / f"cmapss_{dataset_id}.csv"
        frame.to_csv(out_path, index=False)
        outputs[dataset_id] = out_path
    return outputs


def _print_summary(dataset_id: str, frame: pd.DataFrame) -> None:
    n_units = frame["unit"].nunique()
    n_rows = len(frame)
    failure_rate = frame["label"].mean()
    print(
        f"  {dataset_id}: {n_units} units, {n_rows:,} rows, "
        f"failure_rate={failure_rate:.3f}"
    )


def main() -> None:
    output_dir = _cmapss_dir()
    print(f"CMAPSS export → {output_dir}")

    # Strategy 1: check if raw files already exist in data/cmapss
    already_have = all(
        (output_dir / f"cmapss_{did}.csv").exists() for did in DATASET_IDS
    )
    if already_have:
        print("All CMAPSS CSVs already present. Re-reading for summary.")
        for dataset_id in DATASET_IDS:
            frame = pd.read_csv(output_dir / f"cmapss_{dataset_id}.csv")
            _print_summary(dataset_id, frame)
        return

    # Strategy 2: check if raw .txt files are already in data/cmapss
    raw_txts_present = any(
        (output_dir / f"train_{did}.txt").exists() for did in DATASET_IDS
    )
    if raw_txts_present:
        print("Found raw .txt files in data/cmapss — parsing them.")
        outputs = export_cmapss_from_dir(output_dir, output_dir=output_dir)
        for dataset_id, path in outputs.items():
            frame = pd.read_csv(path)
            _print_summary(dataset_id, frame)
        if len(outputs) == len(DATASET_IDS):
            return

    # Strategy 3: download from NASA
    print(f"Attempting download from NASA: {CMAPSS_NASA_URL}")
    try:
        import urllib.request
        with urllib.request.urlopen(CMAPSS_NASA_URL, timeout=60) as response:
            zip_bytes = response.read()
        print(f"  Downloaded {len(zip_bytes):,} bytes.")
        outputs = export_cmapss_from_zip(zip_bytes, output_dir=output_dir)
        for dataset_id, path in outputs.items():
            frame = pd.read_csv(path)
            _print_summary(dataset_id, frame)
        if outputs:
            print("Download and parse succeeded.")
            return
    except Exception as exc:
        print(f"  Download failed: {exc}")

    # Strategy 4: synthetic fallback
    print()
    print("=" * 70)
    print("CMAPSS download unavailable.")
    print()
    print("To use the real dataset, download the zip manually from either:")
    print(f"  {CMAPSS_NASA_URL}")
    print("  https://www.kaggle.com/datasets/behrad3d/nasa-cmapss")
    print()
    print("Then unzip it so the files train_FD001.txt ... train_FD004.txt")
    print(f"are in:  {output_dir}")
    print("and re-run this script.")
    print()
    print("Generating synthetic fallback (5 units × ~200 cycles, drifting sensors)…")
    print("=" * 70)

    outputs = export_cmapss_synthetic_fallback(output_dir=output_dir)
    for dataset_id, path in outputs.items():
        frame = pd.read_csv(path)
        _print_summary(dataset_id, frame)
    print()
    print(
        "Synthetic data written. The benchmark runner will work end-to-end,\n"
        "but the 'frozen model degrades' signal will be weaker than with real data."
    )


if __name__ == "__main__":
    main()
