from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ..tabular_benchmark import TabularBatch, _build_reference_batches, build_tabular_stream
from ..tabular_benchmark import _build_real_tabular_source
from ..torch_model import TorchTabularAdapterModel
from .loader import ReplayRecord, ReplayStream
from ..runtime.model_adapter import SklearnModelAdapter, TorchTabularModelAdapter, clone_model_adapter


@dataclass(frozen=True)
class RealDataBundle:
    """Trained model + replay stream for a real public dataset."""

    source_id: str
    wedge: str
    description: str
    adapter_kind: str
    feature_dim: int
    train_size: int
    stream_size: int
    stream: ReplayStream
    build_layer: Callable[..., object]
    reference_batches: list[TabularBatch]
    validation_accuracy: float
    dataset_path: str | None = None


def _fraud_data_dir() -> Path:
    from ..workspace import fraud_data_dir

    return fraud_data_dir()


def _resolve_ieee_cis_csv_path(csv_path: str | Path | None = None) -> Path:
    if csv_path is not None:
        return Path(csv_path)
    fraud_dir = _fraud_data_dir()
    preferred = fraud_dir / "ieee_cis_full.csv"
    if preferred.exists():
        return preferred
    return fraud_dir / "ieee_cis_sample.csv"


def _gas_sensor_data_dir() -> Path:
    from ..workspace import data_dir

    return data_dir() / "uci_gas_sensor_drift" / "raw" / "Dataset"


def _controller_profile_for_wedge(wedge: str) -> str:
    if wedge == "fraud_risk":
        return "fraud"
    if wedge == "predictive_maintenance":
        return "sensor"
    return "general"


def _tag_stream_metadata(
    stream: ReplayStream,
    *,
    wedge: str,
    controller_profile: str | None = None,
) -> ReplayStream:
    profile = controller_profile or _controller_profile_for_wedge(wedge)
    records = tuple(
        ReplayRecord(
            timestamp=record.timestamp,
            features=record.features,
            label=record.label,
            metadata={
                **record.metadata,
                "wedge": wedge,
                "controller_profile": profile,
            },
        )
        for record in stream.records
    )
    return ReplayStream(records=records, feature_columns=stream.feature_columns)


def _records_from_matrix(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    regimes: list[str] | None = None,
    start_time: datetime | None = None,
) -> tuple[ReplayRecord, ...]:
    start = start_time or datetime(2025, 1, 1)
    records: list[ReplayRecord] = []
    for index, (row, label) in enumerate(zip(features, labels)):
        timestamp = (start + timedelta(minutes=index)).isoformat() + "Z"
        regime = regimes[index] if regimes is not None else "live"
        records.append(
            ReplayRecord(
                timestamp=timestamp,
                features=row.astype(np.float32),
                label=int(label),
                metadata={"row_index": index, "regime": regime},
            )
        )
    return tuple(records)


def _build_natural_batch_stream(
    batch_frames: list[tuple[str, np.ndarray, np.ndarray]],
) -> ReplayStream:
    records: list[ReplayRecord] = []
    feature_dim = batch_frames[0][1].shape[1]
    global_index = 0
    for batch_index, (batch_name, features, labels) in enumerate(batch_frames):
        for row_index in range(len(labels)):
            records.append(
                ReplayRecord(
                    timestamp=f"2025-07-{(batch_index % 28) + 1:02d}T{row_index % 24:02d}:{global_index % 60:02d}:00Z",
                    features=features[row_index].astype(np.float32),
                    label=int(labels[row_index]),
                    metadata={
                        "regime": batch_name,
                        "batch_index": batch_index,
                        "row_index": row_index,
                        "time_ordered": True,
                    },
                )
            )
            global_index += 1
    return ReplayStream(
        records=tuple(records),
        feature_columns=tuple(f"feature_{index}" for index in range(feature_dim)),
    )


def _shift_features(features: np.ndarray, regime: str, step: int, rng: np.random.Generator) -> np.ndarray:
    transformed = features.copy()
    if regime == "stable":
        return transformed
    if regime == "covariate_drift":
        strength = min(1.0, step / 12.0)
        transformed[:, : min(8, transformed.shape[1])] *= 1.0 + 0.35 * strength
        transformed += rng.normal(0.0, 0.05 + 0.05 * strength, size=transformed.shape)
        return transformed
    if regime == "label_shift":
        transformed[:, : min(6, transformed.shape[1])] *= 1.2
        return transformed
    if regime == "abrupt_shift":
        transformed[:, : min(10, transformed.shape[1])] = np.tanh(1.5 * transformed[:, : min(10, transformed.shape[1])])
        return transformed
    transformed *= 1.1
    transformed += rng.normal(0.0, 0.04, size=transformed.shape)
    return transformed


def _build_shifted_stream_from_pool(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    stream_cycles: int = 1,
) -> ReplayStream:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(y_pool == 1)
    negative = np.flatnonzero(y_pool == 0)
    records: list[ReplayRecord] = []
    step_index = 0
    total_steps = steps * max(1, stream_cycles)
    for step in range(total_steps):
        local_step = step % steps
        if local_step < steps // 4:
            regime = "stable"
            positive_rate = 0.55
        elif local_step < steps // 2:
            regime = "covariate_drift"
            positive_rate = 0.55
        elif local_step < 3 * steps // 4:
            regime = "label_shift"
            positive_rate = 0.78
        else:
            regime = "abrupt_shift"
            positive_rate = 0.35

        positive_count = int(round(batch_size * positive_rate))
        negative_count = batch_size - positive_count
        indices = np.concatenate(
            [
                rng.choice(positive, positive_count, replace=True),
                rng.choice(negative, negative_count, replace=True),
            ]
        )
        rng.shuffle(indices)
        batch_x = _shift_features(x_pool[indices], regime, local_step, rng)
        batch_y = y_pool[indices]
        batch_records = _records_from_matrix(
            batch_x,
            batch_y,
            regimes=[regime] * batch_size,
        )
        for record in batch_records:
            records.append(
                ReplayRecord(
                    timestamp=f"2025-01-{(step % 28) + 1:02d}T{step:02d}:{step_index % 60:02d}:00Z",
                    features=record.features,
                    label=record.label,
                    metadata={"regime": regime, "step": step, "cycle": step // steps},
                )
            )
            step_index += 1

    feature_dim = x_pool.shape[1]
    return ReplayStream(
        records=tuple(records),
        feature_columns=tuple(f"feature_{index}" for index in range(feature_dim)),
    )


def _regime_for_local_step(local_step: int, steps: int) -> str:
    if local_step < steps // 4:
        return "stable"
    if local_step < steps // 2:
        return "covariate_drift"
    if local_step < 3 * steps // 4:
        return "label_shift"
    return "abrupt_shift"


def _build_chronological_shifted_stream(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    time_rank: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    seed: int,
    stream_cycles: int = 1,
    apply_synthetic_shift: bool = True,
) -> ReplayStream:
    """Time-ordered stream with optional synthetic regime perturbations."""

    order = np.argsort(time_rank)
    x_sorted = x_pool[order]
    y_sorted = y_pool[order]
    pool_size = len(y_sorted)
    rng = np.random.default_rng(seed)
    records: list[ReplayRecord] = []
    step_index = 0
    total_steps = steps * max(1, stream_cycles)

    for step in range(total_steps):
        local_step = step % steps
        regime = _regime_for_local_step(local_step, steps)
        start = (step * batch_size) % max(1, pool_size)
        indices = np.arange(start, start + batch_size) % pool_size
        base_batch_x = x_sorted[indices]
        batch_x = (
            _shift_features(base_batch_x, regime, local_step, rng)
            if apply_synthetic_shift
            else base_batch_x.copy()
        )
        batch_y = y_sorted[indices]
        for row_index in range(batch_size):
            records.append(
                ReplayRecord(
                    timestamp=f"2025-06-{(step % 28) + 1:02d}T{step:02d}:{row_index:02d}:00Z",
                    features=batch_x[row_index],
                    label=int(batch_y[row_index]),
                    metadata={
                        "regime": regime,
                        "step": step,
                        "cycle": step // steps,
                        "time_ordered": True,
                    },
                )
            )
            step_index += 1

    feature_dim = x_pool.shape[1]
    return ReplayStream(
        records=tuple(records),
        feature_columns=tuple(f"feature_{index}" for index in range(feature_dim)),
    )


def _append_temporal_context_features(
    features: np.ndarray,
    time_rank: np.ndarray,
) -> np.ndarray:
    order = np.argsort(time_rank)
    sorted_features = features[order].astype(np.float64)
    sorted_time = time_rank[order].astype(np.float64)
    deltas = np.diff(sorted_time, prepend=sorted_time[0])
    positive_deltas = deltas[deltas > 0]
    median_delta = float(np.median(positive_deltas)) if len(positive_deltas) > 0 else 1.0
    short_ema = sorted_features[0].copy()
    long_ema = sorted_features[0].copy()
    context = np.zeros((len(sorted_features), 8), dtype=np.float32)
    denom = max(1, len(sorted_features) - 1)
    for index, row in enumerate(sorted_features):
        normalized_time = float(index) / float(denom)
        log_gap = float(np.log1p(max(0.0, deltas[index]) / max(median_delta, 1e-6)))
        short_abs_dev = float(np.mean(np.abs(row - short_ema)))
        long_abs_dev = float(np.mean(np.abs(row - long_ema)))
        short_rms_dev = float(np.sqrt(np.mean((row - short_ema) ** 2)))
        long_rms_dev = float(np.sqrt(np.mean((row - long_ema) ** 2)))
        row_mean = float(np.mean(row))
        trend = float(np.mean(short_ema - long_ema))
        context[index] = np.array(
            [
                normalized_time,
                log_gap,
                row_mean,
                float(np.std(row)),
                short_abs_dev,
                long_abs_dev,
                short_rms_dev,
                trend,
            ],
            dtype=np.float32,
        )
        short_ema = 0.85 * short_ema + 0.15 * row
        long_ema = 0.97 * long_ema + 0.03 * row

    augmented = np.zeros((len(features), features.shape[1] + context.shape[1]), dtype=np.float32)
    augmented[order, : context.shape[1]] = context
    augmented[order, context.shape[1] :] = sorted_features.astype(np.float32)
    return augmented


def _load_fraud_feature_frame(
    csv_path: Path,
    *,
    augment_temporal_context: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    if "label" not in frame.columns:
        raise ValueError(f"{csv_path} must contain a label column")
    feature_cols = [column for column in frame.columns if column.startswith("feature_")]
    if not feature_cols:
        raise ValueError(f"{csv_path} has no feature_* columns")
    features = frame[feature_cols].to_numpy(dtype=np.float32)
    labels = frame["label"].astype(np.int64).to_numpy()
    if "time_rank" in frame.columns:
        time_rank = frame["time_rank"].astype(np.int64).to_numpy()
    elif "TransactionDT" in frame.columns:
        time_rank = frame["TransactionDT"].astype(np.int64).to_numpy()
    elif "step" in frame.columns:
        time_rank = frame["step"].astype(np.int64).to_numpy()
    else:
        time_rank = np.arange(len(labels), dtype=np.int64)
    if augment_temporal_context:
        features = _append_temporal_context_features(features, time_rank)
    return features, labels, time_rank


def _load_gas_sensor_batches(
    *,
    target_class: int = 2,
    data_dir: str | Path | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    root = Path(data_dir) if data_dir is not None else _gas_sensor_data_dir()
    paths = sorted(root.glob("batch*.dat"), key=lambda path: int(path.stem.replace("batch", "")))
    if not paths:
        raise FileNotFoundError(f"No gas sensor batch files found under {root}")

    max_feature_index = 0
    cached_rows: list[tuple[str, list[tuple[int, float]], int]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if not parts:
                    continue
                label = int(parts[0])
                feature_pairs: list[tuple[int, float]] = []
                for token in parts[1:]:
                    index_text, value_text = token.split(":", 1)
                    feature_index = int(index_text) - 1
                    max_feature_index = max(max_feature_index, feature_index + 1)
                    feature_pairs.append((feature_index, float(value_text)))
                cached_rows.append((path.stem, feature_pairs, label))

    grouped_features: dict[str, list[np.ndarray]] = {}
    grouped_labels: dict[str, list[int]] = {}
    for batch_name, feature_pairs, label in cached_rows:
        row = np.zeros(max_feature_index, dtype=np.float32)
        for feature_index, value in feature_pairs:
            row[feature_index] = value
        grouped_features.setdefault(batch_name, []).append(row)
        grouped_labels.setdefault(batch_name, []).append(1 if label == target_class else 0)

    result: list[tuple[str, np.ndarray, np.ndarray]] = []
    for path in paths:
        batch_name = path.stem
        features = np.vstack(grouped_features[batch_name]).astype(np.float32)
        labels = np.asarray(grouped_labels[batch_name], dtype=np.int64)
        result.append((batch_name, features, labels))
    return result


def _split_train_test_indices(
    labels: np.ndarray,
    time_rank: np.ndarray,
    *,
    test_fraction: float,
    seed: int,
    temporal_split: bool,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(labels))
    if temporal_split:
        order = np.argsort(time_rank)
        split = max(1, int(len(order) * (1.0 - test_fraction)))
        if split >= len(order):
            split = len(order) - 1
        return order[:split], order[split:]
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split = int(len(indices) * (1.0 - test_fraction))
    return indices[:split], indices[split:]


def _restrict_stream_pool_tail(
    features: np.ndarray,
    labels: np.ndarray,
    time_rank: np.ndarray,
    *,
    tail_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only the most recent tail of a chronologically ordered test pool."""

    fraction = float(np.clip(tail_fraction, 0.05, 1.0))
    if fraction >= 0.999:
        return features, labels, time_rank
    order = np.argsort(time_rank)
    start = max(0, int(len(order) * (1.0 - fraction)))
    chosen = order[start:]
    return features[chosen], labels[chosen], time_rank[chosen]


def _train_sklearn_fraud_bundle(
    *,
    source_id: str,
    description: str,
    csv_path: Path,
    steps: int,
    batch_size: int,
    seed: int,
    stream_cycles: int = 1,
    test_fraction: float = 0.25,
    stream_tail_fraction: float = 1.0,
    apply_synthetic_shift: bool = True,
    temporal_split: bool = False,
) -> RealDataBundle:
    features, labels, time_rank = _load_fraud_feature_frame(csv_path)
    train_idx, test_idx = _split_train_test_indices(
        labels,
        time_rank,
        test_fraction=test_fraction,
        seed=seed,
        temporal_split=temporal_split,
    )
    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    time_test = time_rank[test_idx]
    x_test, y_test, time_test = _restrict_stream_pool_tail(
        x_test,
        y_test,
        time_test,
        tail_fraction=stream_tail_fraction,
    )

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=0.20,
        random_state=seed + 1,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )

    estimator = LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed)
    estimator.fit(x_train, y_train)
    adapter = SklearnModelAdapter(
        estimator,
        model_version=f"{source_id}-v1",
        source_feature_mean=x_train.mean(axis=0),
        source_feature_std=np.clip(x_train.std(axis=0), 1e-3, None),
        source_positive_rate=float(y_train.mean()),
    )
    validation_accuracy = float((adapter.predict(x_validation) == y_validation).mean())
    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )
    stream = _build_chronological_shifted_stream(
        x_test,
        y_test,
        time_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed + 3,
        stream_cycles=stream_cycles,
        apply_synthetic_shift=apply_synthetic_shift,
    )

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches

        return build_reliability_layer_from_reference_batches(
            clone_model_adapter(adapter),
            reference_batches,
            config=config,
        )

    return RealDataBundle(
        source_id=source_id,
        wedge="fraud_risk",
        description=description,
        adapter_kind="sklearn",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(stream.records),
        stream=_tag_stream_metadata(stream, wedge="fraud_risk"),
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=validation_accuracy,
        dataset_path=str(csv_path.resolve()),
    )


def load_paysim_fraud_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 7,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
    apply_synthetic_shift: bool = True,
    temporal_split: bool = False,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path else _fraud_data_dir() / "paysim.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"PaySim CSV not found at {path}. Run: python3 scripts/export_bundled_fraud_data.py"
        )
    return _train_sklearn_fraud_bundle(
        source_id="paysim_fraud",
        description="PaySim-inspired synthetic mobile-money fraud (time-ordered stream with regime feature shift).",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        apply_synthetic_shift=apply_synthetic_shift,
        temporal_split=temporal_split,
    )


def load_ieee_cis_fraud_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 11,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
    apply_synthetic_shift: bool = False,
    temporal_split: bool = False,
) -> RealDataBundle:
    path = _resolve_ieee_cis_csv_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"IEEE-CIS CSV not found at {path}. Run: python3 scripts/export_bundled_fraud_data.py"
        )
    description = (
        "IEEE-CIS fraud full processed bundle (chronological replay from raw Kaggle transaction data)."
        if path.name == "ieee_cis_full.csv"
        else "IEEE-CIS-like fraud sample (Kaggle export or synthetic fallback), chronological replay."
    )
    return _train_sklearn_fraud_bundle(
        source_id="ieee_cis_fraud",
        description=description,
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        apply_synthetic_shift=apply_synthetic_shift,
        temporal_split=temporal_split,
    )


def load_ulb_creditcard_fraud_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 13,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
    apply_synthetic_shift: bool = False,
    temporal_split: bool = False,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "creditcard.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"ULB credit card CSV not found at {path}. Run: python3 scripts/export_open_datasets.py"
        )
    return _train_sklearn_fraud_bundle(
        source_id="ulb_creditcard_fraud",
        description="ULB credit card fraud (OpenML) — chronological transaction stream with natural drift.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        apply_synthetic_shift=apply_synthetic_shift,
        temporal_split=temporal_split,
    )


def _train_torch_fraud_bundle(
    *,
    source_id: str,
    description: str,
    csv_path: Path,
    steps: int,
    batch_size: int,
    seed: int,
    stream_cycles: int = 1,
    epochs: int = 12,
    temporal_split: bool = False,
    test_fraction: float = 0.25,
    stream_tail_fraction: float = 1.0,
    stream_apply_synthetic_shift: bool = True,
    augment_temporal_context: bool = False,
) -> RealDataBundle:
    features, labels, time_rank = _load_fraud_feature_frame(
        csv_path,
        augment_temporal_context=augment_temporal_context,
    )
    train_idx, test_idx = _split_train_test_indices(
        labels,
        time_rank,
        test_fraction=test_fraction,
        seed=seed,
        temporal_split=temporal_split,
    )
    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    time_test = time_rank[test_idx]
    x_test, y_test, time_test = _restrict_stream_pool_tail(
        x_test,
        y_test,
        time_test,
        tail_fraction=stream_tail_fraction,
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train,
        y_train,
        test_size=0.20,
        random_state=seed + 1,
        stratify=y_train if len(np.unique(y_train)) > 1 else None,
    )
    return _train_torch_bundle(
        source_id=source_id,
        wedge="fraud_risk",
        description=description,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_test,
        y_stream_pool=y_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        epochs=epochs,
        stream_cycles=stream_cycles,
        stream_time_ranks=time_test,
        stream_apply_synthetic_shift=stream_apply_synthetic_shift,
    )


def load_paysim_fraud_torch_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 7,
    stream_cycles: int = 1,
    temporal_split: bool = False,
) -> RealDataBundle:
    path = _fraud_data_dir() / "paysim.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run python3 scripts/export_bundled_fraud_data.py first ({path})")
    return _train_torch_fraud_bundle(
        source_id="paysim_fraud_torch",
        description="PaySim fraud with torch adapter for full-intervention research comparison.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=temporal_split,
    )


def load_ieee_cis_fraud_torch_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 11,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
    temporal_split: bool = False,
) -> RealDataBundle:
    path = _resolve_ieee_cis_csv_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Run python3 scripts/export_bundled_fraud_data.py first ({path})")
    description = (
        "IEEE-CIS fraud full processed bundle with torch adapter for chronological intervention testing."
        if path.name == "ieee_cis_full.csv"
        else "IEEE-CIS-like fraud sample with torch adapter for chronological intervention testing."
    )
    return _train_torch_fraud_bundle(
        source_id="ieee_cis_fraud_torch",
        description=description,
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=temporal_split,
    )


def load_ulb_creditcard_fraud_torch_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 13,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
    temporal_split: bool = False,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "creditcard.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run python3 scripts/export_open_datasets.py first ({path})")
    return _train_torch_fraud_bundle(
        source_id="ulb_creditcard_fraud_torch",
        description="ULB credit card fraud with torch adapter for full-intervention SOTA comparison.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=temporal_split,
    )


def load_ieee_cis_fraud_torch_hard_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 11,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
) -> RealDataBundle:
    """Train on the earliest 50% of time; replay only the latest 50% of the holdout tail."""

    path = _resolve_ieee_cis_csv_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Run python3 scripts/export_bundled_fraud_data.py first ({path})")
    description = (
        "IEEE-CIS hard temporal slice: 50/50 train-test split, stream = latest half of holdout."
        if path.name == "ieee_cis_full.csv"
        else "IEEE-CIS-like hard temporal slice with torch adapter."
    )
    return _train_torch_fraud_bundle(
        source_id="ieee_cis_fraud_torch_hard",
        description=description,
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=True,
        test_fraction=0.50,
        stream_tail_fraction=0.50,
    )


def load_ieee_cis_fraud_torch_context_hard_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 11,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
) -> RealDataBundle:
    path = _resolve_ieee_cis_csv_path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Run python3 scripts/export_bundled_fraud_data.py first ({path})")
    description = (
        "IEEE-CIS hard temporal slice with appended temporal-context features."
        if path.name == "ieee_cis_full.csv"
        else "IEEE-CIS-like hard temporal slice with appended temporal-context features."
    )
    return _train_torch_fraud_bundle(
        source_id="ieee_cis_fraud_torch_context_hard",
        description=description,
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=True,
        test_fraction=0.50,
        stream_tail_fraction=0.50,
        augment_temporal_context=True,
    )


def load_ulb_creditcard_fraud_torch_hard_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 13,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
) -> RealDataBundle:
    """Train on earliest transactions; replay the latest half of the holdout period."""

    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "creditcard.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run python3 scripts/export_open_datasets.py first ({path})")
    return _train_torch_fraud_bundle(
        source_id="ulb_creditcard_fraud_torch_hard",
        description="ULB credit card hard temporal slice: 50/50 split, latest holdout half only.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=True,
        test_fraction=0.50,
        stream_tail_fraction=0.50,
    )


def load_paysim_fraud_torch_hard_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 7,
    stream_cycles: int = 1,
) -> RealDataBundle:
    """PaySim with harder temporal split and no synthetic feature shift on the stream."""

    path = _fraud_data_dir() / "paysim.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run python3 scripts/export_bundled_fraud_data.py first ({path})")
    return _train_torch_fraud_bundle(
        source_id="paysim_fraud_torch_hard",
        description="PaySim hard temporal slice: 50/50 split, latest holdout half, natural stream only.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=True,
        test_fraction=0.50,
        stream_tail_fraction=0.50,
        stream_apply_synthetic_shift=False,
    )


def load_elliptic_fraud_torch_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 19,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
    temporal_split: bool = True,
    test_fraction: float = 0.25,
    stream_tail_fraction: float = 1.0,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "elliptic_fraud.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Elliptic CSV not found at {path}. Run: python3 scripts/ingest_elliptic_kaggle_zip.py "
            "or python3 scripts/export_elliptic_baf_fraud_data.py (fallback)"
        )
    return _train_torch_fraud_bundle(
        source_id="elliptic_fraud_torch",
        description="Elliptic Bitcoin illicit/licit transactions — temporal tabular replay (labeled nodes only).",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=temporal_split,
        test_fraction=test_fraction,
        stream_tail_fraction=stream_tail_fraction,
        stream_apply_synthetic_shift=False,
    )


def load_elliptic_fraud_torch_hard_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 19,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "elliptic_fraud.csv"
    if not path.exists():
        raise FileNotFoundError(f"Elliptic CSV not found at {path}")
    return _train_torch_fraud_bundle(
        source_id="elliptic_fraud_torch_hard",
        description="Elliptic hard temporal slice: 50/50 split, latest holdout half only.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=True,
        test_fraction=0.50,
        stream_tail_fraction=0.50,
        stream_apply_synthetic_shift=False,
    )


def load_elliptic_fraud_torch_context_hard_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 19,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "elliptic_fraud.csv"
    if not path.exists():
        raise FileNotFoundError(f"Elliptic CSV not found at {path}")
    return _train_torch_fraud_bundle(
        source_id="elliptic_fraud_torch_context_hard",
        description="Elliptic hard temporal slice with appended temporal-context features.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=True,
        test_fraction=0.50,
        stream_tail_fraction=0.50,
        stream_apply_synthetic_shift=False,
        augment_temporal_context=True,
    )


def load_baf_fraud_torch_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 21,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
    temporal_split: bool = True,
    test_fraction: float = 0.25,
    stream_tail_fraction: float = 1.0,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "baf_base_fraud.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"BAF CSV not found at {path}. Run: python3 scripts/ingest_baf_kaggle_zip.py "
            "or python3 scripts/export_elliptic_baf_fraud_data.py (fallback)"
        )
    return _train_torch_fraud_bundle(
        source_id="baf_fraud_torch",
        description="Bank Account Fraud (BAF Base) — month-ordered account-opening fraud applications.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=temporal_split,
        test_fraction=test_fraction,
        stream_tail_fraction=stream_tail_fraction,
        stream_apply_synthetic_shift=False,
    )


def load_baf_fraud_torch_hard_bundle(
    *,
    steps: int = 32,
    batch_size: int = 64,
    seed: int = 21,
    stream_cycles: int = 1,
    csv_path: str | Path | None = None,
) -> RealDataBundle:
    path = Path(csv_path) if csv_path is not None else _fraud_data_dir() / "baf_base_fraud.csv"
    if not path.exists():
        raise FileNotFoundError(f"BAF CSV not found at {path}")
    return _train_torch_fraud_bundle(
        source_id="baf_fraud_torch_hard",
        description="BAF hard temporal slice: 50/50 split, latest holdout half only.",
        csv_path=path,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        temporal_split=True,
        test_fraction=0.50,
        stream_tail_fraction=0.50,
        stream_apply_synthetic_shift=False,
    )


def _train_torch_bundle(
    *,
    source_id: str,
    wedge: str,
    description: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_stream_pool: np.ndarray,
    y_stream_pool: np.ndarray,
    steps: int,
    batch_size: int,
    seed: int,
    epochs: int = 20,
    stream_cycles: int = 1,
    stream_time_ranks: np.ndarray | None = None,
    stream_apply_synthetic_shift: bool = True,
    controller_profile: str | None = None,
) -> RealDataBundle:
    model = TorchTabularAdapterModel(x_train.shape[1], seed=seed)
    summary = model.fit_source(x_train, y_train, x_validation, y_validation, epochs=epochs)
    adapter = TorchTabularModelAdapter(model, model_version=f"{source_id}-v1")
    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )
    if stream_time_ranks is not None:
        stream = _build_chronological_shifted_stream(
            x_stream_pool,
            y_stream_pool,
            stream_time_ranks,
            steps=steps,
            batch_size=batch_size,
            seed=seed + 3,
            stream_cycles=stream_cycles,
            apply_synthetic_shift=stream_apply_synthetic_shift,
        )
    else:
        stream = _build_shifted_stream_from_pool(
            x_stream_pool,
            y_stream_pool,
            steps=steps,
            batch_size=batch_size,
            seed=seed + 3,
            stream_cycles=stream_cycles,
        )

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches

        return build_reliability_layer_from_reference_batches(
            clone_model_adapter(adapter),
            reference_batches,
            config=config,
        )

    return RealDataBundle(
        source_id=source_id,
        wedge=wedge,
        description=description,
        adapter_kind="torch_tabular",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(stream.records),
        stream=_tag_stream_metadata(stream, wedge=wedge, controller_profile=controller_profile),
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=summary.best_validation_accuracy,
    )


def _train_sklearn_bundle(
    *,
    source_id: str,
    wedge: str,
    description: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_stream_pool: np.ndarray,
    y_stream_pool: np.ndarray,
    steps: int,
    batch_size: int,
    seed: int,
    stream_cycles: int = 1,
    stream_time_ranks: np.ndarray | None = None,
    stream_apply_synthetic_shift: bool = True,
    controller_profile: str | None = None,
) -> RealDataBundle:
    estimator = LogisticRegression(max_iter=400, random_state=seed)
    estimator.fit(x_train, y_train)
    adapter = SklearnModelAdapter(
        estimator,
        model_version=f"{source_id}-v1",
        source_feature_mean=x_train.mean(axis=0),
        source_feature_std=np.clip(x_train.std(axis=0), 1e-3, None),
        source_positive_rate=float(y_train.mean()),
    )
    validation_accuracy = float((adapter.predict(x_validation) == y_validation).mean())
    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )
    if stream_time_ranks is not None:
        stream = _build_chronological_shifted_stream(
            x_stream_pool,
            y_stream_pool,
            stream_time_ranks,
            steps=steps,
            batch_size=batch_size,
            seed=seed + 3,
            stream_cycles=stream_cycles,
            apply_synthetic_shift=stream_apply_synthetic_shift,
        )
    else:
        stream = _build_shifted_stream_from_pool(
            x_stream_pool,
            y_stream_pool,
            steps=steps,
            batch_size=batch_size,
            seed=seed + 3,
            stream_cycles=stream_cycles,
        )

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches

        return build_reliability_layer_from_reference_batches(
            clone_model_adapter(adapter),
            reference_batches,
            config=config,
        )

    return RealDataBundle(
        source_id=source_id,
        wedge=wedge,
        description=description,
        adapter_kind="sklearn",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(stream.records),
        stream=_tag_stream_metadata(stream, wedge=wedge, controller_profile=controller_profile),
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=validation_accuracy,
    )


def load_breast_cancer_bundle(*, steps: int = 24, batch_size: int = 32, seed: int = 7) -> RealDataBundle:
    x_train, y_train, x_validation, y_validation, x_test, y_test = _build_real_tabular_source(seed=seed)
    return _train_torch_bundle(
        source_id="sklearn_breast_cancer",
        wedge="general_tabular",
        description="UCI breast cancer Wisconsin dataset with regime-based streaming shift.",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_test,
        y_stream_pool=y_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )


def load_digits_bundle(*, steps: int = 24, batch_size: int = 48, seed: int = 7) -> RealDataBundle:
    digits = load_digits()
    features = StandardScaler().fit_transform(digits.data).astype(np.float32)
    labels = (digits.target >= 5).astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.35, random_state=seed, stratify=labels)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=0.25, random_state=seed + 1, stratify=y_train
    )
    return _train_torch_bundle(
        source_id="sklearn_digits",
        wedge="general_tabular",
        description="Sklearn digits binary task with synthetic streaming covariate and label shift.",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_test,
        y_stream_pool=y_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        epochs=15,
    )


def _load_openml_frame(name: str, version: int):
    from ..workspace import data_dir

    cache_dir = data_dir() / "openml_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_csv = data_dir() / "openml" / f"{name.replace('-', '_')}.csv"
    if local_csv.exists():
        return pd.read_csv(local_csv)
    try:
        return fetch_openml(name=name, version=version, as_frame=True, parser="auto", data_home=str(cache_dir))
    except Exception:
        if local_csv.exists():
            return pd.read_csv(local_csv)
        raise


def load_openml_credit_g_bundle(
    *,
    steps: int = 24,
    batch_size: int = 32,
    seed: int = 7,
    stream_cycles: int = 1,
) -> RealDataBundle:
    loaded = _load_openml_frame("credit-g", 1)
    if isinstance(loaded, pd.DataFrame):
        frame = loaded
        feature_cols = [column for column in frame.columns if column.startswith("feature_")]
        features = frame[feature_cols].to_numpy(dtype=np.float32)
        target = frame["label"].astype(np.int64).to_numpy()
    else:
        frame = loaded
        data = frame.data
        target = (frame.target.astype(str) == "good").astype(np.int64).to_numpy()
        numeric = data.select_dtypes(include=["number"]).fillna(0.0)
        features = StandardScaler().fit_transform(numeric.to_numpy()).astype(np.float32)
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.30, random_state=seed, stratify=target)
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=0.25, random_state=seed + 1, stratify=y_train
    )
    return _train_sklearn_bundle(
        source_id="openml_credit_g",
        wedge="fraud_risk",
        description="OpenML German Credit (credit-g) with streaming shift — fraud/risk adjacent tabular benchmark.",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_test,
        y_stream_pool=y_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
    )


def load_openml_electricity_bundle(
    *,
    steps: int = 24,
    batch_size: int = 48,
    seed: int = 7,
    stream_cycles: int = 1,
) -> RealDataBundle:
    loaded = _load_openml_frame("electricity", 1)
    if isinstance(loaded, pd.DataFrame):
        frame = loaded
        feature_cols = [column for column in frame.columns if column.startswith("feature_")]
        features = frame[feature_cols].to_numpy(dtype=np.float32)
        target = frame["label"].astype(np.int64).to_numpy()
        time_rank = np.arange(len(target), dtype=np.int64)
    else:
        frame = loaded
        data = frame.data
        target = (frame.target.astype(str).str.upper() == "P").astype(np.int64).to_numpy()
        numeric = data.select_dtypes(include=["number"]).fillna(0.0).to_numpy(dtype=np.float32)
        time_rank = np.arange(len(target), dtype=np.int64)
        train_end = int(len(target) * 0.55)
        validation_end = int(len(target) * 0.75)
        scaler = StandardScaler()
        scaler.fit(numeric[:train_end])
        features = scaler.transform(numeric).astype(np.float32)
    train_end = int(len(target) * 0.55)
    validation_end = int(len(target) * 0.75)
    x_train = features[:train_end]
    y_train = target[:train_end]
    x_validation = features[train_end:validation_end]
    y_validation = target[train_end:validation_end]
    x_test = features[validation_end:]
    y_test = target[validation_end:]
    time_test = time_rank[validation_end:]
    return _train_sklearn_bundle(
        source_id="openml_electricity",
        wedge="predictive_maintenance",
        description="OpenML Electricity market dataset — chronological operational stream with conservative sensor-safe runtime profile.",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_test,
        y_stream_pool=y_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        stream_cycles=stream_cycles,
        stream_time_ranks=time_test,
        stream_apply_synthetic_shift=False,
        controller_profile="sensor_safe",
    )


def load_openml_electricity_torch_bundle(
    *,
    steps: int = 24,
    batch_size: int = 48,
    seed: int = 7,
    stream_cycles: int = 1,
    epochs: int = 16,
) -> RealDataBundle:
    loaded = _load_openml_frame("electricity", 1)
    if isinstance(loaded, pd.DataFrame):
        frame = loaded
        feature_cols = [column for column in frame.columns if column.startswith("feature_")]
        features = frame[feature_cols].to_numpy(dtype=np.float32)
        target = frame["label"].astype(np.int64).to_numpy()
        time_rank = np.arange(len(target), dtype=np.int64)
    else:
        frame = loaded
        data = frame.data
        target = (frame.target.astype(str).str.upper() == "P").astype(np.int64).to_numpy()
        numeric = data.select_dtypes(include=["number"]).fillna(0.0).to_numpy(dtype=np.float32)
        time_rank = np.arange(len(target), dtype=np.int64)
        train_end = int(len(target) * 0.55)
        scaler = StandardScaler()
        scaler.fit(numeric[:train_end])
        features = scaler.transform(numeric).astype(np.float32)

    train_end = int(len(target) * 0.55)
    validation_end = int(len(target) * 0.75)
    x_train = features[:train_end]
    y_train = target[:train_end]
    x_validation = features[train_end:validation_end]
    y_validation = target[train_end:validation_end]
    x_test = features[validation_end:]
    y_test = target[validation_end:]
    time_test = time_rank[validation_end:]

    return _train_torch_bundle(
        source_id="openml_electricity_torch",
        wedge="predictive_maintenance",
        description="OpenML Electricity market dataset with chronological torch replay and conservative sensor-safe runtime profile.",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_test,
        y_stream_pool=y_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
        epochs=epochs,
        stream_cycles=stream_cycles,
        stream_time_ranks=time_test,
        stream_apply_synthetic_shift=False,
        controller_profile="sensor_safe",
    )


def load_uci_gas_sensor_drift_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 7,
    stream_cycles: int = 1,
    target_class: int = 4,
    data_dir: str | Path | None = None,
) -> RealDataBundle:
    del steps, batch_size, seed, stream_cycles
    batch_frames = _load_gas_sensor_batches(target_class=target_class, data_dir=data_dir)
    train_frames = batch_frames[:4]
    validation_frame = batch_frames[4]
    stream_frames = batch_frames[5:]

    x_train = np.vstack([features for _, features, _ in train_frames]).astype(np.float32)
    y_train = np.concatenate([labels for _, _, labels in train_frames]).astype(np.int64)
    x_validation = validation_frame[1].astype(np.float32)
    y_validation = validation_frame[2].astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    scaled_stream_frames = [
        (batch_name, scaler.transform(features).astype(np.float32), labels.astype(np.int64))
        for batch_name, features, labels in stream_frames
    ]

    estimator = LogisticRegression(max_iter=500, random_state=7, class_weight="balanced")
    estimator.fit(x_train, y_train)
    adapter = SklearnModelAdapter(
        estimator,
        model_version=f"uci_gas_sensor_drift_class{target_class}-v1",
        source_feature_mean=x_train.mean(axis=0),
        source_feature_std=np.clip(x_train.std(axis=0), 1e-3, None),
        source_positive_rate=float(y_train.mean()),
    )
    validation_accuracy = float((adapter.predict(x_validation) == y_validation).mean())
    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=min(64, len(y_validation)),
        seed=target_class + 17,
    )
    stream = _build_natural_batch_stream(scaled_stream_frames)

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches

        return build_reliability_layer_from_reference_batches(
            clone_model_adapter(adapter),
            reference_batches,
            config=config,
        )

    return RealDataBundle(
        source_id="uci_gas_sensor_drift",
        wedge="predictive_maintenance",
        description=f"UCI Gas Sensor Array Drift, batch-chronological one-vs-rest replay (target gas class {target_class}).",
        adapter_kind="sklearn",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(stream.records),
        stream=_tag_stream_metadata(stream, wedge="predictive_maintenance"),
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=validation_accuracy,
    )


def load_uci_gas_sensor_drift_torch_bundle(
    *,
    steps: int = 24,
    batch_size: int = 64,
    seed: int = 7,
    stream_cycles: int = 1,
    target_class: int = 4,
    data_dir: str | Path | None = None,
    epochs: int = 14,
) -> RealDataBundle:
    del steps, batch_size, seed, stream_cycles
    batch_frames = _load_gas_sensor_batches(target_class=target_class, data_dir=data_dir)
    train_frames = batch_frames[:4]
    validation_frame = batch_frames[4]
    stream_frames = batch_frames[5:]

    x_train = np.vstack([features for _, features, _ in train_frames]).astype(np.float32)
    y_train = np.concatenate([labels for _, _, labels in train_frames]).astype(np.int64)
    x_validation = validation_frame[1].astype(np.float32)
    y_validation = validation_frame[2].astype(np.int64)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_validation = scaler.transform(x_validation).astype(np.float32)
    scaled_stream_frames = [
        (batch_name, scaler.transform(features).astype(np.float32), labels.astype(np.int64))
        for batch_name, features, labels in stream_frames
    ]
    x_stream = np.vstack([features for _, features, _ in scaled_stream_frames]).astype(np.float32)
    y_stream = np.concatenate([labels for _, _, labels in scaled_stream_frames]).astype(np.int64)
    time_ranks = np.concatenate(
        [
            np.full(len(labels), batch_index, dtype=np.int64)
            for batch_index, (_, _, labels) in enumerate(scaled_stream_frames, start=6)
        ]
    )

    bundle = _train_torch_bundle(
        source_id="uci_gas_sensor_drift_torch",
        wedge="predictive_maintenance",
        description=f"UCI Gas Sensor Array Drift torch replay (target gas class {target_class}).",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_stream,
        y_stream_pool=y_stream,
        steps=max(1, len(y_stream) // max(1, len(y_validation))),
        batch_size=min(64, len(y_validation)),
        seed=target_class + 101,
        epochs=epochs,
        stream_cycles=1,
        stream_time_ranks=time_ranks,
        stream_apply_synthetic_shift=False,
    )

    stream = _build_natural_batch_stream(scaled_stream_frames)
    return RealDataBundle(
        source_id=bundle.source_id,
        wedge=bundle.wedge,
        description=bundle.description,
        adapter_kind=bundle.adapter_kind,
        feature_dim=bundle.feature_dim,
        train_size=bundle.train_size,
        stream_size=len(stream.records),
        stream=_tag_stream_metadata(stream, wedge=bundle.wedge),
        build_layer=bundle.build_layer,
        reference_batches=bundle.reference_batches,
        validation_accuracy=bundle.validation_accuracy,
    )


def load_wilds_civilcomments_csv_bundle(
    *,
    csv_path: str | Path = "data/wilds/civilcomments_v1.0/all_data_with_identities.csv",
    steps: int = 18,
    batch_size: int = 64,
    seed: int = 7,
    row_limit: int = 12000,
    tfidf_features: int = 4000,
    svd_dim: int = 64,
) -> RealDataBundle:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"WILDS CivilComments CSV not found: {path}")

    frame = pd.read_csv(path, nrows=row_limit)
    texts = frame["comment_text"].fillna("").astype(str).tolist()
    labels = (frame["toxicity"].astype(float) >= 0.5).astype(np.int64).to_numpy()
    groups = frame["black"].fillna(0.0).astype(float).to_numpy()

    indices = np.arange(len(labels))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_a = int(0.55 * len(indices))
    split_b = int(0.75 * len(indices))
    train_idx = indices[:split_a]
    val_idx = indices[split_a:split_b]
    test_idx = indices[split_b:]

    vectorizer = TfidfVectorizer(max_features=tfidf_features, ngram_range=(1, 2), min_df=2)
    x_train_sparse = vectorizer.fit_transform([texts[i] for i in train_idx])
    x_val_sparse = vectorizer.transform([texts[i] for i in val_idx])
    x_test_sparse = vectorizer.transform([texts[i] for i in test_idx])

    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    x_train = svd.fit_transform(x_train_sparse).astype(np.float32)
    x_validation = svd.transform(x_val_sparse).astype(np.float32)
    x_test = svd.transform(x_test_sparse).astype(np.float32)
    y_train = labels[train_idx]
    y_validation = labels[val_idx]
    y_test = labels[test_idx]
    group_test = groups[test_idx]

    easy = np.flatnonzero(group_test < 0.5)
    hard = np.flatnonzero(group_test >= 0.5)
    if len(easy) == 0 or len(hard) == 0:
        easy = np.arange(len(y_test) // 2)
        hard = np.arange(len(y_test) // 2, len(y_test))

    records: list[ReplayRecord] = []
    schedule = [
        ("easy_stable", easy),
        ("hard_shift", hard),
        ("easy_return", easy),
        ("hard_recurrence", hard),
    ]
    segment = max(1, steps // len(schedule))
    cursor = 0
    for step in range(steps):
        regime, pool = schedule[min(len(schedule) - 1, step // segment)]
        chosen = rng.choice(pool, size=batch_size, replace=len(pool) < batch_size)
        batch_x = x_test[chosen]
        batch_y = y_test[chosen]
        for row_index in range(batch_size):
            records.append(
                ReplayRecord(
                    timestamp=f"2025-02-{(step % 28) + 1:02d}T{step:02d}:{row_index:02d}:00Z",
                    features=batch_x[row_index],
                    label=int(batch_y[row_index]),
                    metadata={"regime": regime, "step": step, "source": "wilds_civilcomments_csv"},
                )
            )
            cursor += 1

    model = TorchTabularAdapterModel(x_train.shape[1], seed=seed)
    summary = model.fit_source(x_train, y_train, x_validation, y_validation, epochs=12)
    adapter = TorchTabularModelAdapter(model, model_version="wilds-civilcomments-v1")
    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=batch_size,
        seed=seed + 17,
    )

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches

        return build_reliability_layer_from_reference_batches(
            clone_model_adapter(adapter),
            reference_batches,
            config=config,
        )

    stream = ReplayStream(
        records=tuple(records),
        feature_columns=tuple(f"feature_{index}" for index in range(x_train.shape[1])),
    )
    return RealDataBundle(
        source_id="wilds_civilcomments_csv",
        wedge="public_nlp",
        description="Local WILDS CivilComments CSV (TF-IDF + SVD) with group-based streaming shift.",
        adapter_kind="torch_tabular",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(stream.records),
        stream=_tag_stream_metadata(stream, wedge="public_nlp"),
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=summary.best_validation_accuracy,
    )


def load_breast_cancer_tabular_stream_bundle(*, steps: int = 18, batch_size: int = 48, seed: int = 7) -> RealDataBundle:
    """Uses the project's existing tabular shift stream builder on breast cancer."""

    x_train, y_train, x_validation, y_validation, x_test, y_test = _build_real_tabular_source(seed=seed)
    batches = build_tabular_stream(x_test, y_test, steps=steps, batch_size=batch_size, seed=seed)
    records: list[ReplayRecord] = []
    for step, batch in enumerate(batches):
        for row_index in range(len(batch.labels)):
            records.append(
                ReplayRecord(
                    timestamp=f"2025-03-{(step % 28) + 1:02d}T{step:02d}:{row_index:02d}:00Z",
                    features=batch.features[row_index],
                    label=int(batch.labels[row_index]),
                    metadata={"regime": batch.regime, "step": step},
                )
            )
    bundle = _train_torch_bundle(
        source_id="tabular_breast_cancer_shift",
        wedge="general_tabular",
        description="In-repo breast cancer tabular shift stream (same generator as research benchmarks).",
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        x_stream_pool=x_test,
        y_stream_pool=y_test,
        steps=steps,
        batch_size=batch_size,
        seed=seed,
    )
    return RealDataBundle(
        source_id=bundle.source_id,
        wedge=bundle.wedge,
        description=bundle.description,
        adapter_kind=bundle.adapter_kind,
        feature_dim=bundle.feature_dim,
        train_size=bundle.train_size,
        stream_size=len(records),
        stream=ReplayStream(records=tuple(records), feature_columns=bundle.stream.feature_columns),
        build_layer=bundle.build_layer,
        reference_batches=bundle.reference_batches,
        validation_accuracy=bundle.validation_accuracy,
    )


def load_credit_macro_shock_bundle(
    *,
    steps: int = 24,
    batch_size: int = 32,
    seed: int = 7,
    shock_magnitude: float = 2.0,
    shock_start_fraction: float = 0.35,
    shock_end_fraction: float = 0.70,
    recovery_fraction: float = 0.85,
) -> RealDataBundle:
    """German Credit with a simulated macro-regime shock (COVID-era proxy).

    The test stream is divided into four phases:

    1. **Normal** (0 → shock_start): standard distribution
    2. **Shock** (shock_start → shock_end): economic feature groups multiplied by
       ``shock_magnitude``, positive rate increased (more defaults)
    3. **Recovery** (shock_end → recovery_fraction): gradual linear return toward normal
    4. **New normal** (recovery → end): slightly shifted baseline

    This tests whether ARL can detect the macro shift and adapt without
    over-fitting to the shocked distribution before recovery.
    """
    loaded = _load_openml_frame("credit-g", 1)
    if isinstance(loaded, pd.DataFrame):
        frame = loaded
        feature_cols = [c for c in frame.columns if c.startswith("feature_")]
        features = frame[feature_cols].to_numpy(dtype=np.float32)
        target = frame["label"].astype(np.int64).to_numpy()
    else:
        frame = loaded
        data = frame.data
        target = (frame.target.astype(str) == "good").astype(np.int64).to_numpy()
        numeric = data.select_dtypes(include=["number"]).fillna(0.0)
        features = StandardScaler().fit_transform(numeric.to_numpy()).astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.35, random_state=seed, stratify=target
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train, y_train, test_size=0.20, random_state=seed + 1, stratify=y_train
    )

    # Build shock stream: step-indexed batches with phase-modulated features
    rng = np.random.default_rng(seed + 42)
    n_test = len(x_test)
    records: list[ReplayRecord] = []

    for step in range(steps):
        progress = step / max(1, steps - 1)
        chosen = rng.integers(0, n_test, size=batch_size)
        x_batch = x_test[chosen].copy()
        y_batch = y_test[chosen].copy()

        if progress < shock_start_fraction:
            regime = "pre_shock"
            multiplier = 1.0
        elif progress < shock_end_fraction:
            regime = "macro_shock"
            phase = (progress - shock_start_fraction) / (shock_end_fraction - shock_start_fraction)
            multiplier = 1.0 + (shock_magnitude - 1.0) * min(1.0, phase * 1.5)
            # Flip some good → bad during shock (higher default rate)
            flip_mask = (y_batch == 1) & (rng.random(batch_size) < 0.25 * phase)
            y_batch = np.where(flip_mask, 0, y_batch).astype(np.int64)
        elif progress < recovery_fraction:
            regime = "recovery"
            phase = (progress - shock_end_fraction) / (recovery_fraction - shock_end_fraction)
            multiplier = shock_magnitude - (shock_magnitude - 1.0) * phase
        else:
            regime = "new_normal"
            multiplier = 1.05  # slightly shifted from original

        # Apply multiplier to first half of features (economic proxies)
        half = x_batch.shape[1] // 2
        x_batch[:, :half] *= float(multiplier)

        for i in range(batch_size):
            records.append(ReplayRecord(
                timestamp=f"2020-{(step % 12) + 1:02d}-{(i % 28) + 1:02d}T00:00:00Z",
                features=x_batch[i],
                label=int(y_batch[i]),
                metadata={"regime": regime, "step": step, "shock_multiplier": float(multiplier)},
            ))

    estimator = LogisticRegression(max_iter=400, random_state=seed)
    estimator.fit(x_train, y_train)
    adapter = SklearnModelAdapter(
        estimator,
        model_version="credit_macro_shock-v1",
        source_feature_mean=x_train.mean(axis=0),
        source_feature_std=np.clip(x_train.std(axis=0), 1e-3, None),
        source_positive_rate=float(y_train.mean()),
    )
    reference_batches = _build_reference_batches(x_validation, y_validation, batch_size=batch_size, seed=seed + 17)
    stream = ReplayStream(records=tuple(records), feature_columns=tuple(f"feature_{i}" for i in range(x_train.shape[1])))

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches
        return build_reliability_layer_from_reference_batches(clone_model_adapter(adapter), reference_batches, config=config)

    return RealDataBundle(
        source_id="credit_macro_shock",
        wedge="fraud_risk",
        description=f"German Credit with simulated macro shock (×{shock_magnitude}) then recovery — COVID-era proxy.",
        adapter_kind="sklearn",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(records),
        stream=stream,
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=float((adapter.predict(x_validation) == y_validation).mean()),
    )


def load_wilds_civilcomments_neural_bundle(
    *,
    csv_path: str | Path = "data/wilds/civilcomments_v1.0/all_data_with_identities.csv",
    steps: int = 18,
    batch_size: int = 64,
    seed: int = 7,
    row_limit: int = 8000,
    embed_dim: int = 64,
    model_name: str = "all-MiniLM-L6-v2",
) -> RealDataBundle:
    """WILDS CivilComments with neural text embeddings (sentence-transformers if available).

    Falls back to TF-IDF + SVD when ``sentence_transformers`` is not installed,
    so the bundle always works.  When neural embeddings are used, the feature
    representation is higher quality and the shift patterns are more realistic.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"WILDS CivilComments CSV not found: {path}")

    frame = pd.read_csv(path, nrows=row_limit)
    texts = frame["comment_text"].fillna("").astype(str).tolist()
    labels = (frame["toxicity"].astype(float) >= 0.5).astype(np.int64).to_numpy()
    groups = frame["black"].fillna(0.0).astype(float).to_numpy()

    # Try sentence-transformers; fall back to TF-IDF + SVD
    neural_encoder_used = False
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(model_name)
        embeddings = encoder.encode(texts, show_progress_bar=False, batch_size=64, convert_to_numpy=True)
        features = embeddings.astype(np.float32)
        # Reduce to embed_dim with PCA for consistency
        if features.shape[1] > embed_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=embed_dim, random_state=seed)
            features = pca.fit_transform(features).astype(np.float32)
        neural_encoder_used = True
    except ImportError:
        vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=2)
        sparse = vectorizer.fit_transform(texts)
        svd = TruncatedSVD(n_components=embed_dim, random_state=seed)
        features = svd.fit_transform(sparse).astype(np.float32)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    split_a = int(0.55 * len(indices))
    split_b = int(0.75 * len(indices))
    train_idx = indices[:split_a]
    val_idx = indices[split_a:split_b]
    test_idx = indices[split_b:]

    x_train, y_train = features[train_idx], labels[train_idx]
    x_validation, y_validation = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]
    group_test = groups[test_idx]

    easy = np.flatnonzero(group_test < 0.5)
    hard = np.flatnonzero(group_test >= 0.5)
    if len(easy) == 0 or len(hard) == 0:
        easy = np.arange(len(y_test) // 2)
        hard = np.arange(len(y_test) // 2, len(y_test))

    schedule = [("easy_stable", easy), ("hard_shift", hard), ("easy_return", easy), ("hard_recurrence", hard)]
    segment = max(1, steps // len(schedule))
    records: list[ReplayRecord] = []
    for step in range(steps):
        regime, pool = schedule[min(len(schedule) - 1, step // segment)]
        chosen = rng.choice(pool, size=batch_size, replace=len(pool) < batch_size)
        for i in range(batch_size):
            records.append(ReplayRecord(
                timestamp=f"2025-02-{(step % 28) + 1:02d}T{step:02d}:{i:02d}:00Z",
                features=x_test[chosen[i]],
                label=int(y_test[chosen[i]]),
                metadata={"regime": regime, "step": step, "neural_encoder": neural_encoder_used},
            ))

    model = TorchTabularAdapterModel(x_train.shape[1], seed=seed)
    summary = model.fit_source(x_train, y_train, x_validation, y_validation, epochs=12)
    adapter = TorchTabularModelAdapter(model, model_version="wilds-civilcomments-neural-v1")
    reference_batches = _build_reference_batches(x_validation, y_validation, batch_size=batch_size, seed=seed + 17)
    stream = ReplayStream(records=tuple(records), feature_columns=tuple(f"feature_{i}" for i in range(x_train.shape[1])))
    encoder_tag = f"sentence-transformers/{model_name}" if neural_encoder_used else "tfidf+svd"

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches
        return build_reliability_layer_from_reference_batches(clone_model_adapter(adapter), reference_batches, config=config)

    return RealDataBundle(
        source_id="wilds_civilcomments_neural",
        wedge="public_nlp",
        description=f"WILDS CivilComments with {encoder_tag} embeddings, group-based streaming shift.",
        adapter_kind="torch_tabular",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(records),
        stream=stream,
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=summary.best_validation_accuracy,
    )


def load_tweeteval_bundle(
    *,
    steps: int = 20,
    batch_size: int = 48,
    seed: int = 7,
    subset: str = "hate",
    embed_dim: int = 64,
) -> RealDataBundle:
    """TweetEval concept drift benchmark.

    Loads the TweetEval dataset (hate speech or offensive subset) via
    HuggingFace ``datasets`` if available.  Falls back to a synthetic
    tweet-like stream with adversarial vocabulary shift when not installed.

    The stream simulates concept drift by alternating between:
    - Phase A: "standard" toxic language patterns
    - Phase B: "evolved" slang and coded language (distribution shift)
    - Phase C: mix of both (co-occurrence drift)
    """
    try:
        from datasets import load_dataset as _hf_load_dataset
        ds = _hf_load_dataset("tweet_eval", subset, trust_remote_code=False)
        train_texts = [x["text"] for x in ds["train"]]
        train_labels = np.array([x["label"] for x in ds["train"]], dtype=np.int64)
        test_texts = [x["text"] for x in ds["test"]]
        test_labels = np.array([x["label"] for x in ds["test"]], dtype=np.int64)
        train_labels = (train_labels > 0).astype(np.int64)
        test_labels = (test_labels > 0).astype(np.int64)
        hf_available = True
    except (ImportError, Exception):
        # Synthetic fallback: token-count features simulating vocabulary shift
        rng_synth = np.random.default_rng(seed)
        n_train, n_test = 2400, 800
        n_features = 50
        train_features_raw = rng_synth.standard_normal((n_train, n_features)).astype(np.float32)
        train_labels = (train_features_raw[:, 0] + rng_synth.standard_normal(n_train) * 0.5 > 0).astype(np.int64)
        test_features_raw = rng_synth.standard_normal((n_test, n_features)).astype(np.float32)
        test_labels = (test_features_raw[:, 0] + rng_synth.standard_normal(n_test) * 0.5 > 0).astype(np.int64)
        # Simulate adversarial drift in second half
        mid = n_test // 2
        test_features_raw[mid:, :10] += 2.5
        hf_available = False

        x_train = StandardScaler().fit_transform(train_features_raw).astype(np.float32)
        x_test = test_features_raw

        x_train, x_validation, y_train, y_validation = train_test_split(
            x_train, train_labels, test_size=0.20, random_state=seed, stratify=train_labels
        )
        return _build_tweeteval_bundle_from_arrays(
            x_train, y_train, x_validation, y_validation, x_test, test_labels,
            steps=steps, batch_size=batch_size, seed=seed, hf_available=hf_available,
        )

    # Encode with TF-IDF + SVD (or sentence-transformers if available)
    all_texts = train_texts + test_texts
    try:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("all-MiniLM-L6-v2")
        all_embeddings = encoder.encode(all_texts, show_progress_bar=False, batch_size=64, convert_to_numpy=True)
        all_features = all_embeddings.astype(np.float32)
        if all_features.shape[1] > embed_dim:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=embed_dim, random_state=seed)
            all_features = pca.fit_transform(all_features).astype(np.float32)
    except ImportError:
        vec = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)
        sparse = vec.fit_transform(all_texts)
        svd = TruncatedSVD(n_components=embed_dim, random_state=seed)
        all_features = svd.fit_transform(sparse).astype(np.float32)

    n_train_total = len(train_texts)
    train_features = all_features[:n_train_total]
    test_features = all_features[n_train_total:]

    scaler = StandardScaler().fit(train_features)
    x_train_all = scaler.transform(train_features).astype(np.float32)
    x_test = scaler.transform(test_features).astype(np.float32)

    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_all, train_labels, test_size=0.15, random_state=seed, stratify=train_labels
    )
    return _build_tweeteval_bundle_from_arrays(
        x_train, y_train, x_validation, y_validation, x_test, test_labels,
        steps=steps, batch_size=batch_size, seed=seed, hf_available=hf_available,
    )


def _build_tweeteval_bundle_from_arrays(
    x_train, y_train, x_validation, y_validation, x_test, y_test,
    *, steps, batch_size, seed, hf_available,
) -> RealDataBundle:
    rng = np.random.default_rng(seed)
    n_test = len(x_test)
    records: list[ReplayRecord] = []

    # Phase schedule: stable → shifted → mixed → recurrence
    phases = [
        ("stable", slice(0, n_test // 3)),
        ("concept_shifted", slice(n_test // 3, 2 * n_test // 3)),
        ("mixed_drift", slice(2 * n_test // 3, n_test)),
    ]
    for step in range(steps):
        regime, pool_slice = phases[min(len(phases) - 1, step * len(phases) // steps)]
        pool_indices = np.arange(n_test)[pool_slice]
        chosen = rng.choice(pool_indices, size=batch_size, replace=len(pool_indices) < batch_size)
        for i in range(batch_size):
            records.append(ReplayRecord(
                timestamp=f"2023-{(step % 12) + 1:02d}-{(i % 28) + 1:02d}T00:00:00Z",
                features=x_test[chosen[i]],
                label=int(y_test[chosen[i]]),
                metadata={"regime": regime, "step": step, "hf_available": hf_available},
            ))

    model = TorchTabularAdapterModel(x_train.shape[1], seed=seed)
    summary = model.fit_source(x_train, y_train, x_validation, y_validation, epochs=10)
    adapter = TorchTabularModelAdapter(model, model_version="tweeteval-v1")
    reference_batches = _build_reference_batches(x_validation, y_validation, batch_size=batch_size, seed=seed + 17)
    stream = ReplayStream(records=tuple(records), feature_columns=tuple(f"feature_{i}" for i in range(x_train.shape[1])))

    def build_layer(config):
        from ..runtime.layer import build_reliability_layer_from_reference_batches
        return build_reliability_layer_from_reference_batches(clone_model_adapter(adapter), reference_batches, config=config)

    source_tag = "tweeteval_hf" if hf_available else "tweeteval_synthetic"
    return RealDataBundle(
        source_id=source_tag,
        wedge="public_nlp",
        description="TweetEval concept drift benchmark (hate/offensive shift across adversarial slang phases).",
        adapter_kind="torch_tabular",
        feature_dim=x_train.shape[1],
        train_size=len(y_train),
        stream_size=len(records),
        stream=stream,
        build_layer=build_layer,
        reference_batches=reference_batches,
        validation_accuracy=summary.best_validation_accuracy,
    )


REAL_DATA_LOADERS: dict[str, Callable[..., RealDataBundle]] = {
    "breast_cancer": load_breast_cancer_bundle,
    "digits": load_digits_bundle,
    "openml_credit_g": load_openml_credit_g_bundle,
    "openml_electricity": load_openml_electricity_bundle,
    "openml_electricity_torch": load_openml_electricity_torch_bundle,
    "uci_gas_sensor_drift": load_uci_gas_sensor_drift_bundle,
    "uci_gas_sensor_drift_torch": load_uci_gas_sensor_drift_torch_bundle,
    "wilds_civilcomments_csv": load_wilds_civilcomments_csv_bundle,
    "wilds_civilcomments_neural": load_wilds_civilcomments_neural_bundle,
    "credit_macro_shock": load_credit_macro_shock_bundle,
    "tweeteval_synthetic": lambda **kw: load_tweeteval_bundle(**kw),
    "tabular_breast_cancer_shift": load_breast_cancer_tabular_stream_bundle,
    "paysim_fraud": load_paysim_fraud_bundle,
    "ieee_cis_fraud": load_ieee_cis_fraud_bundle,
    "ulb_creditcard_fraud": load_ulb_creditcard_fraud_bundle,
    "paysim_fraud_torch": load_paysim_fraud_torch_bundle,
    "ieee_cis_fraud_torch": load_ieee_cis_fraud_torch_bundle,
    "ieee_cis_fraud_torch_hard": load_ieee_cis_fraud_torch_hard_bundle,
    "ieee_cis_fraud_torch_context_hard": load_ieee_cis_fraud_torch_context_hard_bundle,
    "ulb_creditcard_fraud_torch": load_ulb_creditcard_fraud_torch_bundle,
    "ulb_creditcard_fraud_torch_hard": load_ulb_creditcard_fraud_torch_hard_bundle,
    "paysim_fraud_torch_hard": load_paysim_fraud_torch_hard_bundle,
    "elliptic_fraud_torch": load_elliptic_fraud_torch_bundle,
    "elliptic_fraud_torch_hard": load_elliptic_fraud_torch_hard_bundle,
    "elliptic_fraud_torch_context_hard": load_elliptic_fraud_torch_context_hard_bundle,
    "baf_fraud_torch": load_baf_fraud_torch_bundle,
    "baf_fraud_torch_hard": load_baf_fraud_torch_hard_bundle,
}


def load_real_data_bundle(source_id: str, **kwargs) -> RealDataBundle:
    if source_id not in REAL_DATA_LOADERS:
        raise KeyError(f"unknown source_id {source_id!r}; available={sorted(REAL_DATA_LOADERS)}")
    return REAL_DATA_LOADERS[source_id](**kwargs)
