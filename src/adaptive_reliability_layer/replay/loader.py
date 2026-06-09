from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from ..runtime.config import ReplayConfig
from ..runtime.types import RuntimeBatch

CANONICAL_REPLAY_REQUIRED_COLUMNS = ("timestamp", "label")
CANONICAL_REPLAY_OPTIONAL_COLUMNS = ("regime", "sample_id")
CANONICAL_REPLAY_METADATA_PREFIX = "meta_"


@dataclass(frozen=True)
class ReplayRecord:
    timestamp: str
    features: np.ndarray
    label: int | None
    metadata: dict


@dataclass(frozen=True)
class ReplayStream:
    records: tuple[ReplayRecord, ...]
    feature_columns: tuple[str, ...]


def load_replay_table(path: str | Path, config: ReplayConfig) -> ReplayStream:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(data_path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(data_path)
    else:
        raise ValueError(f"unsupported replay table format: {data_path.suffix!r}")

    return _frame_to_stream(frame, config)


def load_replay_csv(path: str | Path, config: ReplayConfig) -> ReplayStream:
    return load_replay_table(path, config)


def render_replay_schema_markdown(feature_prefix: str = "feature_") -> str:
    return "\n".join(
        [
            "Canonical replay schema",
            "",
            "Required columns:",
            f"- `timestamp`",
            f"- `label`",
            f"- `{feature_prefix}0`, `{feature_prefix}1`, ...",
            "",
            "Optional columns:",
            "- `regime`",
            "- `sample_id`",
            f"- `{CANONICAL_REPLAY_METADATA_PREFIX}*` for arbitrary metadata",
            "",
            "CSV and Parquet are both supported.",
        ]
    )


def _frame_to_stream(frame: pd.DataFrame, config: ReplayConfig) -> ReplayStream:
    feature_columns = tuple(
        column
        for column in frame.columns
        if column.startswith(config.feature_prefix) or column.startswith("f_")
    )
    if not feature_columns:
        numeric = frame.select_dtypes(include=["number"]).columns.tolist()
        excluded = {config.label_column, config.timestamp_column}
        feature_columns = tuple(column for column in numeric if column not in excluded)
    if not feature_columns:
        raise ValueError("no feature columns detected in replay CSV")

    records: list[ReplayRecord] = []
    for _, row in frame.iterrows():
        label_value = row.get(config.label_column)
        label = None if pd.isna(label_value) else int(label_value)
        metadata = {"source_row": int(row.name)}
        if "regime" in frame.columns and not pd.isna(row.get("regime")):
            metadata["regime"] = str(row.get("regime"))
        if "sample_id" in frame.columns and not pd.isna(row.get("sample_id")):
            metadata["sample_id"] = str(row.get("sample_id"))
        for column in frame.columns:
            if column.startswith(CANONICAL_REPLAY_METADATA_PREFIX) and not pd.isna(row.get(column)):
                metadata[column] = row.get(column)
        records.append(
            ReplayRecord(
                timestamp=str(row.get(config.timestamp_column, "")),
                features=row[list(feature_columns)].to_numpy(dtype=np.float32),
                label=label,
                metadata=metadata,
            )
        )
    return ReplayStream(records=tuple(records), feature_columns=feature_columns)


def iter_replay_batches(
    stream: ReplayStream,
    *,
    batch_size: int,
    label_delay_steps: int = 0,
    max_steps: int | None = None,
) -> Iterator[tuple[int, RuntimeBatch, int | None]]:
    """Yield (step, batch, delayed_label_step) tuples.

    The returned batch always carries the labels that belong to its own feature rows.
    Delayed supervision is handled by the replay engine rather than by swapping in
    labels from an older batch.
    """

    records = list(stream.records)
    total_steps = len(records) // batch_size
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)

    for step in range(total_steps):
        start = step * batch_size
        end = start + batch_size
        chunk = records[start:end]
        features = np.stack([record.features for record in chunk], axis=0)
        labels = np.array([record.label if record.label is not None else -1 for record in chunk])
        has_labels = bool(np.all(labels >= 0))

        delayed_label_step = step - label_delay_steps if label_delay_steps > 0 else step
        regime_values = [
            str(record.metadata.get("regime", "")).strip()
            for record in chunk
            if record.metadata.get("regime") is not None
        ]
        batch_regime = regime_values[-1] if regime_values else f"replay_step_{step}"
        controller_profiles = [
            str(record.metadata.get("controller_profile", "")).strip()
            for record in chunk
            if record.metadata.get("controller_profile") is not None
        ]
        batch_controller_profile = controller_profiles[-1] if controller_profiles else "general"
        wedges = [
            str(record.metadata.get("wedge", "")).strip()
            for record in chunk
            if record.metadata.get("wedge") is not None
        ]
        batch_wedge = wedges[-1] if wedges else None

        batch = RuntimeBatch(
            features=features,
            labels=labels if has_labels else None,
            regime=batch_regime,
            timestamp=chunk[-1].timestamp,
            metadata={
                "batch_size": batch_size,
                "label_delay_steps": label_delay_steps,
                "regime_id": batch_regime,
                "source_rows": [int(record.metadata.get("source_row", -1)) for record in chunk],
                "controller_profile": batch_controller_profile,
                "wedge": batch_wedge,
            },
        )
        yield step, batch, delayed_label_step
