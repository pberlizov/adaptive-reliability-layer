"""Stable public SDK for external consumers of Adaptive Reliability Layer.

Wraps the production runtime with a simplified interface for the three
common integration patterns: sklearn estimators, PyTorch tabular models,
and black-box prediction functions.

Quick start
-----------
    from adaptive_reliability_layer.sdk import ARLSession, build_session_from_sklearn

    session = build_session_from_sklearn(
        estimator=my_trained_clf,
        reference_features=X_validation,
        reference_labels=y_validation,
    )

    # Score a batch
    result = session.predict(X_batch)
    print(result.predictions, result.shift_score)

    # Later, when labels arrive
    session.reveal(result.batch_id, y_batch)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class PredictResult:
    """Output of a single batch scored through the reliability layer."""

    batch_id: str
    step: int
    predictions: list[int]
    probabilities: list[float]
    confidence: float
    shift_score: float
    action_taken: str
    reliability_score: float
    retrain_recommended: bool
    risk_alert: bool
    label_quality_warnings: list[str] = field(default_factory=list)

    @property
    def positive_rate(self) -> float:
        return float(np.mean(self.predictions)) if self.predictions else 0.0


@dataclass(frozen=True)
class RevealResult:
    """Output of a label-reveal call."""

    batch_id: str
    batch_accuracy: float
    utility: float


class ARLSession:
    """Simplified interface to a running ReliabilityLayer instance.

    All methods are safe to call from a single thread.  For multi-threaded
    serving, use the FastAPI sidecar (``arl-serve``) instead.
    """

    def __init__(self, layer: Any) -> None:
        self._layer = layer
        self._step = 0

    @property
    def layer(self) -> Any:
        """The underlying ReliabilityLayer, for advanced use."""
        return self._layer

    def predict(
        self,
        features: np.ndarray | list,
        *,
        batch_id: str | None = None,
        regime: str = "live",
    ) -> PredictResult:
        """Score a feature batch through the reliability controller.

        Parameters
        ----------
        features : array-like of shape (n_samples, n_features)
        batch_id : optional stable identifier for this batch (used for label reveals)
        regime   : regime hint string (e.g. 'high_risk', 'evening', 'live')

        Returns
        -------
        PredictResult with predictions, shift diagnostics, and action taken.
        """
        import uuid
        from .runtime.types import RuntimeBatch

        feat_array = np.asarray(features, dtype=np.float32)
        if feat_array.ndim == 1:
            feat_array = feat_array.reshape(1, -1)

        auto_id = batch_id or f"sdk-{uuid.uuid4().hex[:8]}"
        batch = RuntimeBatch(
            features=feat_array,
            labels=None,
            regime=regime,
            metadata={"batch_id": auto_id},
        )
        surface = self._layer.process_batch(batch)
        self._step = surface.step
        return PredictResult(
            batch_id=auto_id,
            step=surface.step,
            predictions=surface.predictions,
            probabilities=surface.probabilities,
            confidence=surface.confidence,
            shift_score=surface.shift_score,
            action_taken=surface.action_taken,
            reliability_score=surface.reliability_score,
            retrain_recommended=surface.retrain_recommended,
            risk_alert=surface.risk_alert,
        )

    def reveal(
        self,
        batch_id: str,
        labels: np.ndarray | list[int],
    ) -> RevealResult:
        """Reveal ground-truth labels for a previously scored batch.

        Parameters
        ----------
        batch_id : the batch_id returned by predict()
        labels   : integer array of true labels, same length as the batch

        Returns
        -------
        RevealResult with batch_accuracy and utility.
        """
        label_array = np.asarray(labels, dtype=np.int64)
        metrics = self._layer.reveal_labels_by_batch_id(batch_id, label_array)
        return RevealResult(
            batch_id=batch_id,
            batch_accuracy=float(metrics.get("batch_accuracy", 0.0)),
            utility=float(metrics.get("utility", 0.0)),
        )

    def retrain_needed(self) -> bool:
        """Return True if the controller has signalled that retraining is overdue."""
        if not self._layer.revealed_metrics:
            return False
        recent = list(self._layer.revealed_metrics)[-1]
        return bool(recent.get("retrain_recommended", False))

    def summary(self) -> dict[str, Any]:
        """Return a brief operational summary dict."""
        return {
            "step": self._step,
            "operating_mode": self._layer.config.operating_mode.value,
            "pending_reveals": self._layer.pending_delayed_count,
            "retrain_needed": self.retrain_needed(),
        }


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_session_from_sklearn(
    estimator: Any,
    reference_features: np.ndarray,
    reference_labels: np.ndarray | None = None,
    *,
    operating_mode: str = "bounded_auto",
    label_delay_steps: int = 0,
    audit_dir: str = ".arl",
    model_version: str = "source-v1",
) -> ARLSession:
    """Build an ARLSession wrapping a fitted scikit-learn classifier.

    Parameters
    ----------
    estimator          : a fitted sklearn classifier with predict_proba
    reference_features : holdout features that represent the source distribution
    reference_labels   : holdout labels (optional, improves reference profile)
    operating_mode     : 'shadow', 'recommend', or 'bounded_auto'
    label_delay_steps  : number of batches before labels are available
    audit_dir          : directory for snapshots and audit DB
    model_version      : tag string for audit records
    """
    from dataclasses import replace

    from .runtime.config import GovernanceConfig, MetricsConfig, PolicyConfig, ReplayConfig, RuntimeConfig
    from .runtime.layer import build_reliability_layer_from_reference_batches
    from .runtime.model_adapter import SklearnModelAdapter
    from .runtime.types import OperatingMode
    from .tabular_benchmark import TabularBatch

    feat = np.asarray(reference_features, dtype=np.float32)
    source_mean = feat.mean(axis=0)
    source_std = np.clip(feat.std(axis=0), 1e-3, None)
    positive_rate = (
        float(np.asarray(reference_labels).mean())
        if reference_labels is not None
        else 0.5
    )
    adapter = SklearnModelAdapter(
        estimator,
        model_version=model_version,
        source_feature_mean=source_mean,
        source_feature_std=source_std,
        source_positive_rate=positive_rate,
    )
    batch_size = min(64, max(16, len(feat) // 4))
    ref_batches = [
        TabularBatch(
            features=feat[i : i + batch_size],
            labels=(
                np.asarray(reference_labels[i : i + batch_size], dtype=np.int64)
                if reference_labels is not None
                else np.zeros(min(batch_size, len(feat) - i), dtype=np.int64)
            ),
            regime="reference",
        )
        for i in range(0, len(feat), batch_size)
    ]
    config = RuntimeConfig(
        operating_mode=OperatingMode(operating_mode),
        model_version=model_version,
        governance=GovernanceConfig(
            audit_db_path=f"{audit_dir}/audit.db",
            snapshot_dir=f"{audit_dir}/snapshots",
        ),
        metrics=MetricsConfig(enabled=False),
        replay=ReplayConfig(label_delay_steps=label_delay_steps),
        log_json=False,
    )
    layer = build_reliability_layer_from_reference_batches(adapter, ref_batches, config=config)
    return ARLSession(layer)


def build_session_from_torch(
    model: Any,
    reference_features: np.ndarray,
    reference_labels: np.ndarray | None = None,
    *,
    operating_mode: str = "bounded_auto",
    label_delay_steps: int = 0,
    audit_dir: str = ".arl",
    model_version: str = "source-v1",
) -> ARLSession:
    """Build an ARLSession wrapping a TorchTabularAdapterModel.

    Parameters
    ----------
    model              : a fitted TorchTabularAdapterModel instance
    reference_features : holdout features that represent the source distribution
    reference_labels   : holdout labels (optional)
    """
    from .runtime.config import GovernanceConfig, MetricsConfig, PolicyConfig, ReplayConfig, RuntimeConfig
    from .runtime.layer import build_reliability_layer_from_reference_batches
    from .runtime.model_adapter import TorchTabularModelAdapter
    from .runtime.types import OperatingMode
    from .tabular_benchmark import TabularBatch

    adapter = TorchTabularModelAdapter(model, model_version=model_version)
    feat = np.asarray(reference_features, dtype=np.float32)
    batch_size = min(64, max(16, len(feat) // 4))
    ref_batches = [
        TabularBatch(
            features=feat[i : i + batch_size],
            labels=(
                np.asarray(reference_labels[i : i + batch_size], dtype=np.int64)
                if reference_labels is not None
                else np.zeros(min(batch_size, len(feat) - i), dtype=np.int64)
            ),
            regime="reference",
        )
        for i in range(0, len(feat), batch_size)
    ]
    config = RuntimeConfig(
        operating_mode=OperatingMode(operating_mode),
        model_version=model_version,
        governance=GovernanceConfig(
            audit_db_path=f"{audit_dir}/audit.db",
            snapshot_dir=f"{audit_dir}/snapshots",
        ),
        metrics=MetricsConfig(enabled=False),
        replay=ReplayConfig(label_delay_steps=label_delay_steps),
        log_json=False,
    )
    layer = build_reliability_layer_from_reference_batches(adapter, ref_batches, config=config)
    return ARLSession(layer)


def build_session_from_predict_fn(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    reference_features: np.ndarray,
    *,
    operating_mode: str = "shadow",
    label_delay_steps: int = 0,
    audit_dir: str = ".arl",
    model_version: str = "external-v1",
) -> ARLSession:
    """Build an ARLSession wrapping any predict_proba function (monitor-only).

    The returned session runs in shadow mode by default — it monitors drift
    and fires alerts but cannot modify the model's weights.  Change
    ``operating_mode`` to 'bounded_auto' to enable output correction via
    probability post-processing.

    Parameters
    ----------
    predict_proba_fn : callable(features: ndarray) → ndarray of shape (n, 2) or (n,)
    reference_features : holdout features for reference profile
    """
    from .runtime.config import GovernanceConfig, MetricsConfig, ReplayConfig, RuntimeConfig
    from .runtime.layer import build_reliability_layer_from_reference_batches
    from .runtime.model_adapter import BlackBoxModelAdapter
    from .runtime.types import OperatingMode
    from .tabular_benchmark import TabularBatch

    adapter = BlackBoxModelAdapter(predict_proba_fn, model_version=model_version)
    feat = np.asarray(reference_features, dtype=np.float32)
    batch_size = min(64, max(16, len(feat) // 4))
    ref_batches = [
        TabularBatch(
            features=feat[i : i + batch_size],
            labels=np.zeros(min(batch_size, len(feat) - i), dtype=np.int64),
            regime="reference",
        )
        for i in range(0, len(feat), batch_size)
    ]
    config = RuntimeConfig(
        operating_mode=OperatingMode(operating_mode),
        model_version=model_version,
        governance=GovernanceConfig(
            audit_db_path=f"{audit_dir}/audit.db",
            snapshot_dir=f"{audit_dir}/snapshots",
        ),
        metrics=MetricsConfig(enabled=False),
        replay=ReplayConfig(label_delay_steps=label_delay_steps),
        log_json=False,
    )
    layer = build_reliability_layer_from_reference_batches(adapter, ref_batches, config=config)
    return ARLSession(layer)
