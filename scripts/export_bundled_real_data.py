#!/usr/bin/env python3
"""Export bundled real-data CSV fallbacks for offline verification."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def export_german_credit(root: Path) -> Path:
    raw_path = root / "credit_g.raw"
    if not raw_path.exists():
        raise FileNotFoundError(f"missing {raw_path}")

    frame = pd.read_csv(raw_path, sep=r"\s+", header=None)
    labels = (frame.iloc[:, -1].astype(int) == 1).astype(np.int64).to_numpy()
    features = frame.iloc[:, :-1]
    encoded = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit_transform(features.astype(str))
    scaled = StandardScaler().fit_transform(encoded).astype(np.float32)
    rows = {f"feature_{index}": scaled[:, index] for index in range(scaled.shape[1])}
    rows["label"] = labels
    output = root / "credit_g.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def export_spambase(root: Path) -> Path:
    raw_path = root / "spambase.raw"
    if not raw_path.exists():
        spambase_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
        frame = pd.read_csv(spambase_url, header=None)
    else:
        frame = pd.read_csv(raw_path, header=None)
    labels = frame.iloc[:, -1].astype(np.int64).to_numpy()
    features = StandardScaler().fit_transform(frame.iloc[:, :-1].to_numpy()).astype(np.float32)
    rows = {f"feature_{index}": features[:, index] for index in range(features.shape[1])}
    rows["label"] = labels
    output = root / "electricity.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "data" / "openml"
    root.mkdir(parents=True, exist_ok=True)
    credit = export_german_credit(root)
    electricity = export_spambase(root)
    print(f"Wrote {credit}")
    print(f"Wrote {electricity}")


if __name__ == "__main__":
    main()
