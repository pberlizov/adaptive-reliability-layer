#!/usr/bin/env python3
"""Build ARL-compatible fraud CSVs from Elliptic Bitcoin and BAF (Bank Account Fraud) raw files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _fraud_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "data" / "fraud"
    root.mkdir(parents=True, exist_ok=True)
    return root


def export_elliptic_from_raw(
    *,
    raw_dir: Path | None = None,
    output_name: str = "elliptic_fraud.csv",
    max_features: int = 48,
    max_rows: int | None = None,
) -> Path | None:
    """Merge Elliptic features + class labels into chronological tabular CSV.

    Expects Kaggle raw files:
    - elliptic_txs_features.csv (no header: txId, time_step, feat...)
    - elliptic_txs_classes.csv (txId, class) where class is unknown / 1 (illicit) / 2 (licit)
    """

    root = raw_dir or (_fraud_dir() / "raw" / "elliptic")
    features_path = root / "elliptic_txs_features.csv"
    classes_path = root / "elliptic_txs_classes.csv"
    if not features_path.exists() or not classes_path.exists():
        return None

    feat_df = pd.read_csv(features_path, header=None)
    class_df = pd.read_csv(classes_path)
    if feat_df.shape[1] < 3:
        raise ValueError(f"{features_path} expected at least 3 columns (txId, time_step, features...)")
    if "txId" not in class_df.columns or "class" not in class_df.columns:
        raise ValueError(f"{classes_path} must contain txId and class columns")

    feat_df = feat_df.rename(columns={0: "txId", 1: "time_step"})
    merged = feat_df.merge(class_df, on="txId", how="inner")
    class_map = {"1": 1, 1: 1, "2": 0, 2: 0}
    merged["label"] = merged["class"].map(class_map)
    merged = merged.dropna(subset=["label"])
    merged["label"] = merged["label"].astype(np.int64)
    merged = merged.sort_values(["time_step", "txId"])

    if max_rows is not None and len(merged) > max_rows:
        # Keep timeline head + all illicit in tail window for class balance
        head = merged.iloc[: max_rows // 2]
        tail_pool = merged.iloc[max_rows // 2 :]
        illicit_tail = tail_pool[tail_pool["label"] == 1]
        legit_tail = tail_pool[tail_pool["label"] == 0].head(max(0, max_rows // 2 - len(illicit_tail)))
        merged = pd.concat([head, illicit_tail, legit_tail], ignore_index=True).sort_values(["time_step", "txId"])

    feature_cols = [column for column in merged.columns if isinstance(column, int) or str(column).isdigit()]
    if not feature_cols:
        feature_cols = [column for column in merged.columns if column not in {"txId", "time_step", "class", "label"}]
    numeric = merged[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    if numeric.shape[1] > max_features:
        numeric = numeric[:, :max_features]
    scaled = StandardScaler().fit_transform(numeric).astype(np.float32)
    out = {
        "time_rank": merged["time_step"].astype(np.int64).to_numpy(),
        "label": merged["label"].astype(np.int64).to_numpy(),
    }
    for index in range(scaled.shape[1]):
        out[f"feature_{index}"] = scaled[:, index]
    output = _fraud_dir() / output_name
    pd.DataFrame(out).to_csv(output, index=False)
    return output


def _resolve_baf_variant_csv(raw_dir: Path, variant: str = "Base") -> Path | None:
    candidates = sorted(raw_dir.glob(f"*{variant}*.csv"))
    if not candidates:
        candidates = sorted(raw_dir.glob("*.csv"))
    if not candidates:
        return None
    preferred = [path for path in candidates if variant.lower() in path.stem.lower()]
    return preferred[0] if preferred else candidates[0]


def export_baf_from_raw(
    *,
    raw_dir: Path | None = None,
    variant: str = "Base",
    output_name: str = "baf_base_fraud.csv",
    max_rows: int = 250_000,
    seed: int = 17,
) -> Path | None:
    """Export BAF NeurIPS variant to ARL fraud CSV (month-ordered)."""

    root = raw_dir or (_fraud_dir() / "raw" / "baf")
    source = _resolve_baf_variant_csv(root, variant=variant)
    if source is None:
        return None

    # BAF NeurIPS Base.csv (Feedzai) uses fraud_bool; Kaggle mirrors may use is_fraud.
    label_candidates = ("fraud_bool", "is_fraud", "isFraud", "fraud")
    frame = pd.read_csv(source, low_memory=False)
    label_col = next((column for column in label_candidates if column in frame.columns), None)
    if label_col is None:
        frame = pd.read_csv(source, index_col=0, low_memory=False)
        if "Unnamed: 0" in frame.columns:
            frame = frame.drop(columns=["Unnamed: 0"])
        label_col = next((column for column in label_candidates if column in frame.columns), None)
    if label_col is None:
        preview = ", ".join(map(str, frame.columns[:12]))
        raise ValueError(
            f"{source} missing fraud label column (expected one of {label_candidates}); "
            f"columns start with: {preview}"
        )
    if "month" not in frame.columns:
        frame["month"] = np.arange(len(frame), dtype=np.int64)

    labels = frame[label_col].astype(np.int64).to_numpy()
    time_rank = frame["month"].astype(np.int64).to_numpy()
    excluded = {label_col, "month", "customer_age", "monthly_income"}
    numeric_cols = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not numeric_cols:
        raise ValueError(f"{source} has no numeric feature columns")
    features = frame[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    if len(frame) > max_rows:
        rng = np.random.default_rng(seed)
        order = np.argsort(time_rank)
        frame_idx = order
        head = frame_idx[: max_rows // 2]
        tail_pool = frame_idx[max_rows // 2 :]
        fraud_idx = tail_pool[labels[tail_pool] == 1]
        legit_idx = tail_pool[labels[tail_pool] == 0]
        legit_sample = rng.choice(
            legit_idx,
            size=min(len(legit_idx), max(0, max_rows // 2 - len(fraud_idx))),
            replace=False,
        ) if len(legit_idx) else np.array([], dtype=int)
        chosen = np.concatenate([head, fraud_idx, legit_sample])
        labels = labels[chosen]
        time_rank = time_rank[chosen]
        features = features[chosen]

    order = np.argsort(time_rank)
    labels = labels[order]
    time_rank = time_rank[order]
    features = features[order]
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    out = {"time_rank": time_rank, "label": labels}
    for index in range(scaled.shape[1]):
        out[f"feature_{index}"] = scaled[:, index]
    output = _fraud_dir() / output_name
    pd.DataFrame(out).to_csv(output, index=False)
    return output


def export_elliptic_synthetic_fallback(*, rows: int = 80_000, seed: int = 23) -> Path:
    """Reproducible Elliptic-like stream when Kaggle raw files are unavailable."""

    rng = np.random.default_rng(seed)
    time_rank = np.sort(rng.integers(1, 50, size=rows))
    n_features = 32
    features = rng.normal(size=(rows, n_features)).astype(np.float32)
    fraud_logit = -5.0 + 0.4 * features[:, 0] - 0.25 * features[:, 1] + 0.03 * time_rank.astype(np.float32)
    labels = (rng.random(rows) < (1.0 / (1.0 + np.exp(-fraud_logit)))).astype(np.int64)
    features[labels == 1] += rng.normal(0.6, 0.2, size=(int(labels.sum()), n_features))
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    out = {"time_rank": time_rank, "label": labels}
    for index in range(scaled.shape[1]):
        out[f"feature_{index}"] = scaled[:, index]
    output = _fraud_dir() / "elliptic_fraud.csv"
    pd.DataFrame(out).to_csv(output, index=False)
    return output


def export_baf_synthetic_fallback(*, rows: int = 120_000, seed: int = 29) -> Path:
    """Reproducible BAF-like monthly stream when Kaggle raw files are unavailable."""

    rng = np.random.default_rng(seed)
    months = rng.integers(1, 9, size=rows)
    time_rank = np.argsort(months * 10_000 + rng.integers(0, 10_000, size=rows))
    months = months[np.argsort(time_rank)]
    n_features = 30
    features = rng.normal(size=(rows, n_features)).astype(np.float32)
    fraud_logit = -4.5 + 0.35 * features[:, 2] + 0.15 * months.astype(np.float32)
    labels = (rng.random(rows) < (1.0 / (1.0 + np.exp(-fraud_logit)))).astype(np.int64)
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    out = {"time_rank": months.astype(np.int64), "label": labels}
    for index in range(scaled.shape[1]):
        out[f"feature_{index}"] = scaled[:, index]
    output = _fraud_dir() / "baf_base_fraud.csv"
    pd.DataFrame(out).to_csv(output, index=False)
    return output


if __name__ == "__main__":
    elliptic = export_elliptic_synthetic_fallback()
    baf = export_baf_synthetic_fallback()
    print(f"elliptic: {elliptic}")
    print(f"baf: {baf}")
