from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..runtime.layer import ReliabilityLayer
from ..runtime.types import DeploymentSurface, RuntimeBatch
from .config import ServingConfig


@dataclass
class ServingState:
    """HTTP-side idempotency and batch validation."""

    layer: ReliabilityLayer
    serving: ServingConfig
    idempotent_responses: dict[str, dict] = field(default_factory=dict)

    def process_batch(self, batch: RuntimeBatch) -> dict:
        batch_id = (batch.metadata or {}).get("batch_id")
        if batch_id is not None:
            batch_id = str(batch_id)
            if not self.serving.allow_duplicate_batch_id and batch_id in self.idempotent_responses:
                cached = dict(self.idempotent_responses[batch_id])
                cached["idempotent_replay"] = True
                return cached
            if self.layer.pending_delayed_count >= self.serving.max_pending_batches:
                raise RuntimeError("pending delayed batch queue is full")

        features = np.asarray(batch.features, dtype=np.float32)
        if features.ndim == 1:
            features = features.reshape(1, -1)
        if features.shape[0] > self.serving.max_batch_rows:
            raise ValueError(f"batch exceeds max_batch_rows={self.serving.max_batch_rows}")
        if features.shape[1] > self.serving.max_feature_dim:
            raise ValueError(
                f"feature dimension exceeds max_feature_dim={self.serving.max_feature_dim}, got {features.shape[1]}"
            )
        expected = getattr(self.layer, "_expected_feature_dim", None)
        if expected is not None and features.shape[1] != expected:
            raise ValueError(f"expected feature_dim={expected}, got {features.shape[1]}")

        surface: DeploymentSurface = self.layer.process_batch(batch)
        record = surface.decision_record()
        record["predictions"] = surface.predictions
        record["probabilities"] = surface.probabilities
        record["shift_score"] = surface.shift_score
        if batch_id is not None:
            record["batch_id"] = batch_id
            if not self.serving.allow_duplicate_batch_id:
                self.idempotent_responses[batch_id] = dict(record)
        return record

    def reveal_labels(self, *, batch_id: str | None, step: int | None, labels: np.ndarray) -> dict:
        warnings = _check_label_quality(labels)
        result: dict
        if batch_id is not None:
            result = self.layer.reveal_labels_by_batch_id(batch_id, labels)
        elif step is not None:
            result = self.layer.reveal_labels(step, labels)
        else:
            raise ValueError("batch_id or step is required")
        if warnings:
            result = dict(result)
            result["label_quality_warnings"] = warnings
        return result


def _check_label_quality(labels: np.ndarray) -> list[str]:
    """Detect common label quality issues at the reveal boundary.

    Returns a list of warning strings (empty = clean).  Issues detected:
    - All-zero labels (possible missing positive class)
    - All-one labels (possible labelling error / anomalous batch)
    - NaN or Inf values
    - Labels outside {0, 1} for binary classification
    - Extreme class imbalance (positive rate < 1% or > 99%)
    """
    if labels is None or len(labels) == 0:
        return ["empty_label_array"]
    arr = np.asarray(labels, dtype=np.float64)
    warnings: list[str] = []
    if np.any(~np.isfinite(arr)):
        warnings.append("non_finite_labels")
        return warnings  # can't safely compute other stats on NaN/Inf
    unique = set(arr.tolist())
    if not unique.issubset({0.0, 1.0}):
        warnings.append(f"out_of_range_labels:{sorted(unique - {0.0, 1.0})[:3]}")
    positive_rate = float(arr.mean())
    if positive_rate == 0.0:
        warnings.append("all_negative_labels")
    elif positive_rate == 1.0:
        warnings.append("all_positive_labels")
    elif positive_rate < 0.01:
        warnings.append(f"extreme_class_imbalance_low:{positive_rate:.4f}")
    elif positive_rate > 0.99:
        warnings.append(f"extreme_class_imbalance_high:{positive_rate:.4f}")
    return warnings
