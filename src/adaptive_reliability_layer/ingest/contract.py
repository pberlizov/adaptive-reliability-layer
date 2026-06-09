from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..replay.loader import ReplayRecord, ReplayStream


@dataclass(frozen=True)
class IngestEvent:
    """Canonical streaming event for ARL sidecar ingest."""

    event_id: str
    timestamp: str
    features: np.ndarray
    label: int | None = None
    regime_id: str = "live"
    metadata: dict[str, Any] | None = None

    def to_replay_record(self) -> ReplayRecord:
        meta = dict(self.metadata or {})
        meta.setdefault("event_id", self.event_id)
        meta.setdefault("regime_id", self.regime_id)
        return ReplayRecord(
            timestamp=self.timestamp,
            features=self.features.astype(np.float32),
            label=self.label,
            metadata=meta,
        )


def events_to_replay_stream(events: Iterable[IngestEvent]) -> ReplayStream:
    records = [event.to_replay_record() for event in events]
    if not records:
        return ReplayStream(records=tuple(), feature_columns=tuple())
    feature_dim = records[0].features.shape[0]
    columns = tuple(f"feature_{index}" for index in range(feature_dim))
    return ReplayStream(records=tuple(records), feature_columns=columns)


def load_events_csv(
    path: str | Path,
    *,
    event_id_column: str = "event_id",
    timestamp_column: str = "timestamp",
    label_column: str = "label",
    regime_column: str = "regime_id",
    feature_prefix: str = "feature_",
) -> list[IngestEvent]:
    frame = pd.read_csv(path)
    feature_cols = [column for column in frame.columns if column.startswith(feature_prefix)]
    if not feature_cols:
        numeric = frame.select_dtypes(include=["number"]).columns.tolist()
        excluded = {label_column, event_id_column}
        feature_cols = [column for column in numeric if column not in excluded]

    events: list[IngestEvent] = []
    for _, row in frame.iterrows():
        label_value = row.get(label_column)
        label = None if pd.isna(label_value) else int(label_value)
        events.append(
            IngestEvent(
                event_id=str(row.get(event_id_column, row.name)),
                timestamp=str(row.get(timestamp_column, datetime.now(UTC).isoformat())),
                features=row[feature_cols].to_numpy(dtype=np.float32),
                label=label,
                regime_id=str(row.get(regime_column, "live")),
                metadata={"source_row": int(row.name)},
            )
        )
    return events


def load_events_jsonl(path: str | Path) -> list[IngestEvent]:
    events: list[IngestEvent] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        features = np.asarray(payload["features"], dtype=np.float32)
        label = payload.get("label")
        events.append(
            IngestEvent(
                event_id=str(payload["event_id"]),
                timestamp=str(payload.get("timestamp", datetime.utcnow().isoformat())),
                features=features,
                label=None if label is None else int(label),
                regime_id=str(payload.get("regime_id", "live")),
                metadata=payload.get("metadata"),
            )
        )
    return events
