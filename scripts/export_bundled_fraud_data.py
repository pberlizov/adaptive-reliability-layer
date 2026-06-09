#!/usr/bin/env python3
"""Bundle PaySim-style and IEEE-CIS-style fraud CSVs for offline replay (no Kaggle required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _repo_fraud_dir() -> Path:
    root = Path(__file__).resolve().parents[1] / "data" / "fraud"
    root.mkdir(parents=True, exist_ok=True)
    return root


def export_paysim_synthetic(*, rows: int = 28000, seed: int = 7) -> Path:
    """PaySim-inspired synthetic mobile-money fraud stream (public, reproducible)."""

    rng = np.random.default_rng(seed)
    steps = rng.integers(1, 744, size=rows)
    amounts = rng.lognormal(mean=4.0, sigma=1.35, size=rows).astype(np.float32)
    old_balance = rng.lognormal(mean=8.0, sigma=1.1, size=rows).astype(np.float32)
    new_balance = np.maximum(0.0, old_balance - amounts * rng.uniform(0.0, 1.2, size=rows)).astype(np.float32)
    transfer_type = rng.integers(0, 5, size=rows)
    dest_balance = rng.lognormal(mean=7.5, sigma=1.0, size=rows).astype(np.float32)

    fraud_logit = (
        -7.2
        + 0.45 * np.log1p(amounts)
        + 0.003 * steps
        + 0.8 * (transfer_type == 2).astype(np.float32)
        + 0.5 * (new_balance < 1.0).astype(np.float32)
    )
    fraud_prob = 1.0 / (1.0 + np.exp(-fraud_logit))
    labels = (rng.random(rows) < fraud_prob).astype(np.int64)

    raw = np.stack(
        [
            amounts,
            old_balance,
            new_balance,
            dest_balance,
            steps.astype(np.float32),
            transfer_type.astype(np.float32),
            np.abs(old_balance - new_balance - amounts),
            np.log1p(amounts),
        ],
        axis=1,
    )
    scaled = StandardScaler().fit_transform(raw).astype(np.float32)
    frame = {
        "time_rank": steps.astype(np.int64),
        "label": labels,
    }
    for index in range(scaled.shape[1]):
        frame[f"feature_{index}"] = scaled[:, index]
    output = _repo_fraud_dir() / "paysim.csv"
    pd.DataFrame(frame).sort_values("time_rank").to_csv(output, index=False)
    return output


def export_ieee_from_raw(*, max_rows: int = 40000, seed: int = 11) -> Path | None:
    raw = _repo_fraud_dir() / "raw" / "train_transaction.csv"
    if not raw.exists():
        return None

    frame = pd.read_csv(raw)
    if "TransactionDT" not in frame.columns or "isFraud" not in frame.columns:
        raise ValueError(f"{raw} missing TransactionDT or isFraud columns")

    frame = frame.sort_values("TransactionDT")
    if len(frame) > max_rows:
        rng = np.random.default_rng(seed)
        # Keep early timeline + random tail sample for class balance
        head = frame.iloc[: max_rows // 2]
        tail_pool = frame.iloc[max_rows // 2 :]
        fraud_tail = tail_pool[tail_pool["isFraud"] == 1]
        legit_tail = tail_pool[tail_pool["isFraud"] == 0].sample(
            n=min(len(tail_pool) - len(fraud_tail), max_rows // 2),
            random_state=seed,
        )
        frame = pd.concat([head, fraud_tail, legit_tail], ignore_index=True).sort_values("TransactionDT")

    labels = frame["isFraud"].astype(np.int64)
    time_rank = frame["TransactionDT"].astype(np.int64)
    numeric = frame.select_dtypes(include=["number"]).drop(columns=["isFraud", "TransactionDT"], errors="ignore")
    numeric = numeric.replace([np.inf, -np.inf], np.nan).fillna(-999.0)
    if numeric.shape[1] > 32:
        numeric = numeric.iloc[:, :32]
    scaled = StandardScaler().fit_transform(numeric.to_numpy()).astype(np.float32)
    out_frame = {"time_rank": time_rank.to_numpy(), "label": labels.to_numpy()}
    for index in range(scaled.shape[1]):
        out_frame[f"feature_{index}"] = scaled[:, index]
    output = _repo_fraud_dir() / "ieee_cis_sample.csv"
    pd.DataFrame(out_frame).to_csv(output, index=False)
    return output


def export_ieee_full_from_raw(
    *,
    seed: int = 11,
    max_features: int = 48,
    output_name: str = "ieee_cis_full.csv",
) -> Path | None:
    """Build a fuller chronological IEEE-CIS bundle from raw Kaggle transaction/identity files."""

    fraud_dir = _repo_fraud_dir()
    raw_dir = fraud_dir / "raw"
    transactions = raw_dir / "train_transaction.csv"
    if not transactions.exists():
        return None

    frame = pd.read_csv(transactions, low_memory=False)
    identity = raw_dir / "train_identity.csv"
    if identity.exists():
        identity_frame = pd.read_csv(identity, low_memory=False)
        if "TransactionID" in frame.columns and "TransactionID" in identity_frame.columns:
            frame = frame.merge(identity_frame, how="left", on="TransactionID")

    if "TransactionDT" not in frame.columns or "isFraud" not in frame.columns:
        raise ValueError(f"{transactions} missing TransactionDT or isFraud columns")

    frame = frame.sort_values("TransactionDT").reset_index(drop=True)
    labels = frame["isFraud"].astype(np.int64).to_numpy()
    time_rank = frame["TransactionDT"].astype(np.int64).to_numpy()

    numeric = frame.select_dtypes(include=["number"]).drop(
        columns=["isFraud", "TransactionDT", "TransactionID"],
        errors="ignore",
    )
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    coverage = 1.0 - numeric.isna().mean(axis=0)
    spread = numeric.std(axis=0, skipna=True).fillna(0.0)
    score = coverage * (spread + 1e-6)
    ranked = score.sort_values(ascending=False)
    keep = list(ranked.index[:max_features])
    selected = numeric[keep].fillna(numeric[keep].median()).fillna(-999.0)
    scaled = StandardScaler().fit_transform(selected.to_numpy()).astype(np.float32)

    out_frame = {"time_rank": time_rank, "label": labels}
    for index in range(scaled.shape[1]):
        out_frame[f"feature_{index}"] = scaled[:, index]
    output = fraud_dir / output_name
    pd.DataFrame(out_frame).to_csv(output, index=False)
    return output


def export_ieee_synthetic(*, rows: int = 32000, seed: int = 11) -> Path:
    """IEEE-CIS-like highly imbalanced fraud tabular (fallback when Kaggle CSV absent)."""

    rng = np.random.default_rng(seed)
    time_rank = np.sort(rng.integers(0, 1_000_000, size=rows))
    n_features = 32
    features = rng.normal(size=(rows, n_features)).astype(np.float32)
    fraud_logit = -3.8 + 0.6 * features[:, 0] - 0.4 * features[:, 1] + 0.000002 * time_rank.astype(np.float32)
    labels = (rng.random(rows) < (1.0 / (1.0 + np.exp(-fraud_logit)))).astype(np.int64)
    features[labels == 1] += rng.normal(0.9, 0.35, size=(int(labels.sum()), n_features))

    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    frame = {"time_rank": time_rank, "label": labels}
    for index in range(scaled.shape[1]):
        frame[f"feature_{index}"] = scaled[:, index]
    output = _repo_fraud_dir() / "ieee_cis_sample.csv"
    pd.DataFrame(frame).to_csv(output, index=False)
    return output


def main() -> None:
    paysim = export_paysim_synthetic()
    ieee_full = export_ieee_full_from_raw()
    ieee = export_ieee_from_raw()
    if ieee is None:
        ieee = export_ieee_synthetic()
        print(f"No Kaggle IEEE file at data/fraud/raw/train_transaction.csv — wrote synthetic fallback.")
    print(f"Wrote {paysim}")
    if ieee_full is not None:
        print(f"Wrote {ieee_full}")
    print(f"Wrote {ieee}")


if __name__ == "__main__":
    main()
