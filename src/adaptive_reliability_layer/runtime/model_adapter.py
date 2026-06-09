from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
import pickle

import numpy as np

from ..torch_model import ModelSnapshot, TorchTabularAdapterModel


@dataclass(frozen=True)
class AdapterSnapshot:
    adapter_kind: str
    payload: Any
    version: str = "1"


@runtime_checkable
class ModelAdapter(Protocol):
    """Commercial integration surface for arbitrary deployed models."""

    adapter_kind: str
    supports_adaptation: bool
    model_version: str

    def predict_proba(self, features: np.ndarray) -> list[float]:
        ...

    def predict(self, features: np.ndarray) -> list[int]:
        ...

    def export_snapshot(self) -> AdapterSnapshot:
        ...

    def load_snapshot(self, snapshot: AdapterSnapshot) -> None:
        ...

    def reset(self) -> None:
        ...

    def parameter_drift(self) -> float:
        ...

    def refresh_batch_norm(self, features: np.ndarray, passes: int = 1) -> None:
        ...

    def recalibrate_temperature(
        self,
        *,
        reference_confidence: float,
        observed_confidence: float,
        momentum: float = 0.25,
    ) -> None:
        ...

    def apply_label_shift_correction(
        self,
        *,
        source_positive_rate: float,
        target_positive_rate: float,
        momentum: float = 0.35,
    ) -> None:
        ...

    def apply_covariate_refresh(
        self,
        *,
        features: np.ndarray,
        reference_confidence: float,
        observed_confidence: float,
        intensity: int = 2,
    ) -> None:
        ...

    def apply_latent_recenter(
        self,
        *,
        features: np.ndarray,
        momentum: float = 0.12,
    ) -> None:
        ...

    def adapt(
        self,
        features: np.ndarray,
        probabilities: list[float],
        *,
        learning_rate: float,
        confidence_threshold: float,
        anchor_strength: float,
        entropy_weight: float,
        max_parameter_drift: float,
        steps: int = 2,
    ) -> float:
        ...


class TorchTabularModelAdapter:
    """Wraps the research PyTorch tabular model for production runtime use."""

    adapter_kind = "torch_tabular"

    def __init__(self, model: TorchTabularAdapterModel, *, model_version: str = "source-v1") -> None:
        self._model = model
        self.model_version = model_version
        self.supports_adaptation = True

    @property
    def inner(self) -> TorchTabularAdapterModel:
        return self._model

    def predict_proba(self, features: np.ndarray) -> list[float]:
        return self._model.predict_proba(features)

    def predict(self, features: np.ndarray) -> list[int]:
        probabilities = self.predict_proba(features)
        return [1 if probability >= 0.5 else 0 for probability in probabilities]

    def export_snapshot(self) -> AdapterSnapshot:
        snapshot: ModelSnapshot = self._model.export_state()
        return AdapterSnapshot(adapter_kind=self.adapter_kind, payload=snapshot, version=self.model_version)

    def load_snapshot(self, snapshot: AdapterSnapshot) -> None:
        if snapshot.adapter_kind != self.adapter_kind:
            raise ValueError(f"expected adapter {self.adapter_kind}, got {snapshot.adapter_kind}")
        self._model.load_state(snapshot.payload)

    def reset(self) -> None:
        self._model.reset()

    def parameter_drift(self) -> float:
        return self._model.parameter_drift()

    def refresh_batch_norm(self, features: np.ndarray, passes: int = 1) -> None:
        self._model.refresh_batch_norm(features, passes=passes)

    def recalibrate_temperature(
        self,
        *,
        reference_confidence: float,
        observed_confidence: float,
        momentum: float = 0.25,
    ) -> None:
        self._model.recalibrate_temperature(
            reference_confidence=reference_confidence,
            observed_confidence=observed_confidence,
            momentum=momentum,
        )

    def apply_label_shift_correction(
        self,
        *,
        source_positive_rate: float,
        target_positive_rate: float,
        momentum: float = 0.35,
    ) -> None:
        self._model.apply_label_shift_correction(
            source_positive_rate=source_positive_rate,
            target_positive_rate=target_positive_rate,
            momentum=momentum,
        )

    def apply_covariate_refresh(
        self,
        *,
        features: np.ndarray,
        reference_confidence: float,
        observed_confidence: float,
        intensity: int = 2,
    ) -> None:
        passes = max(1, intensity + 1)
        self._model.refresh_batch_norm(features, passes=passes)
        self._model.recalibrate_temperature(
            reference_confidence=reference_confidence,
            observed_confidence=observed_confidence,
            momentum=min(0.28, 0.10 + 0.05 * intensity),
        )

    def apply_latent_recenter(self, features: np.ndarray, *, momentum: float = 0.12) -> None:
        self._model.apply_latent_recenter(features, momentum=momentum)

    def adapt(
        self,
        features: np.ndarray,
        probabilities: list[float],
        *,
        learning_rate: float,
        confidence_threshold: float,
        anchor_strength: float,
        entropy_weight: float,
        max_parameter_drift: float,
        steps: int = 2,
    ) -> float:
        return self._model.adapt(
            features,
            probabilities,
            learning_rate=learning_rate,
            confidence_threshold=confidence_threshold,
            anchor_strength=anchor_strength,
            entropy_weight=entropy_weight,
            max_parameter_drift=max_parameter_drift,
            steps=steps,
        )


class SklearnModelAdapter:
    """Wraps a fitted scikit-learn classifier with monitor-only adaptation hooks."""

    adapter_kind = "sklearn"

    def __init__(
        self,
        estimator: Any,
        *,
        model_version: str = "source-v1",
        source_feature_mean: np.ndarray | None = None,
        source_feature_std: np.ndarray | None = None,
        source_positive_rate: float = 0.5,
    ) -> None:
        self._estimator = estimator
        self._source_estimator = self._clone_estimator(estimator)
        self.model_version = model_version
        self.supports_adaptation = False
        self._temperature = 1.0
        self._bias_offset = 0.0
        self._source_feature_mean = None if source_feature_mean is None else np.asarray(source_feature_mean, dtype=np.float32)
        self._source_feature_std = None if source_feature_std is None else np.clip(
            np.asarray(source_feature_std, dtype=np.float32),
            1e-3,
            None,
        )
        self._running_feature_mean = None if self._source_feature_mean is None else self._source_feature_mean.copy()
        self._running_feature_std = None if self._source_feature_std is None else self._source_feature_std.copy()
        self._feature_correction_strength = 0.0
        self._source_positive_rate = float(source_positive_rate)

    @staticmethod
    def _clone_estimator(estimator: Any) -> Any:
        import pickle

        return pickle.loads(pickle.dumps(estimator))

    def predict_proba(self, features: np.ndarray) -> list[float]:
        transformed = self._transform_features(np.asarray(features, dtype=np.float32))
        raw = self._estimator.predict_proba(transformed)
        if raw.shape[1] == 1:
            positive = raw[:, 0]
        else:
            positive = raw[:, 1]
        logits = np.log(np.clip(positive, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - positive, 1e-6, 1.0 - 1e-6))
        calibrated = 1.0 / (1.0 + np.exp(-((logits + self._bias_offset) / self._temperature)))
        return calibrated.tolist()

    def predict(self, features: np.ndarray) -> list[int]:
        return [1 if probability >= 0.5 else 0 for probability in self.predict_proba(features)]

    def export_snapshot(self) -> AdapterSnapshot:
        import pickle

        return AdapterSnapshot(
            adapter_kind=self.adapter_kind,
            payload={
                "estimator": pickle.dumps(self._estimator),
                "temperature": self._temperature,
                "bias_offset": self._bias_offset,
                "source_feature_mean": self._source_feature_mean,
                "source_feature_std": self._source_feature_std,
                "running_feature_mean": self._running_feature_mean,
                "running_feature_std": self._running_feature_std,
                "feature_correction_strength": self._feature_correction_strength,
                "source_positive_rate": self._source_positive_rate,
            },
            version=self.model_version,
        )

    def load_snapshot(self, snapshot: AdapterSnapshot) -> None:
        import pickle

        if snapshot.adapter_kind != self.adapter_kind:
            raise ValueError(f"expected adapter {self.adapter_kind}, got {snapshot.adapter_kind}")
        self._estimator = pickle.loads(snapshot.payload["estimator"])
        self._temperature = float(snapshot.payload["temperature"])
        self._bias_offset = float(snapshot.payload["bias_offset"])
        self._source_feature_mean = snapshot.payload.get("source_feature_mean")
        self._source_feature_std = snapshot.payload.get("source_feature_std")
        self._running_feature_mean = snapshot.payload.get("running_feature_mean")
        self._running_feature_std = snapshot.payload.get("running_feature_std")
        self._feature_correction_strength = float(snapshot.payload.get("feature_correction_strength", 0.0))
        self._source_positive_rate = float(snapshot.payload.get("source_positive_rate", 0.5))

    def reset(self) -> None:
        self._estimator = self._clone_estimator(self._source_estimator)
        self._temperature = 1.0
        self._bias_offset = 0.0
        self._running_feature_mean = None if self._source_feature_mean is None else self._source_feature_mean.copy()
        self._running_feature_std = None if self._source_feature_std is None else self._source_feature_std.copy()
        self._feature_correction_strength = 0.0

    def parameter_drift(self) -> float:
        return abs(self._temperature - 1.0) + abs(self._bias_offset) + self._feature_correction_strength

    def refresh_batch_norm(self, features: np.ndarray, passes: int = 1) -> None:
        if self._source_feature_mean is None or self._source_feature_std is None:
            return
        batch = np.asarray(features, dtype=np.float32)
        batch_mean = batch.mean(axis=0)
        batch_std = np.clip(batch.std(axis=0), 1e-3, None)
        momentum = min(0.85, 0.20 + 0.10 * max(1, passes))
        if self._running_feature_mean is None or self._running_feature_std is None:
            self._running_feature_mean = batch_mean
            self._running_feature_std = batch_std
        else:
            self._running_feature_mean = (
                (1.0 - momentum) * self._running_feature_mean + momentum * batch_mean
            ).astype(np.float32)
            self._running_feature_std = np.clip(
                (1.0 - momentum) * self._running_feature_std + momentum * batch_std,
                1e-3,
                None,
            ).astype(np.float32)
        self._feature_correction_strength = min(0.85, self._feature_correction_strength + 0.18 * max(1, passes))

    def recalibrate_temperature(
        self,
        *,
        reference_confidence: float,
        observed_confidence: float,
        momentum: float = 0.25,
        min_temperature: float = 0.65,
        max_temperature: float = 1.35,
    ) -> None:
        gap = reference_confidence - observed_confidence
        target = min(max(1.0 - 1.4 * gap, min_temperature), max_temperature)
        self._temperature = (1.0 - momentum) * self._temperature + momentum * target
        self._temperature = min(max(self._temperature, min_temperature), max_temperature)

    def apply_label_shift_correction(
        self,
        *,
        source_positive_rate: float,
        target_positive_rate: float,
        momentum: float = 0.35,
        max_abs_bias: float = 1.25,
    ) -> None:
        source = min(max(source_positive_rate, 1e-4), 1.0 - 1e-4)
        target = min(max(target_positive_rate, 1e-4), 1.0 - 1e-4)
        source_logit = float(np.log(source / (1.0 - source)))
        target_logit = float(np.log(target / (1.0 - target)))
        desired = min(max(target_logit - source_logit, -max_abs_bias), max_abs_bias)
        self._bias_offset = (1.0 - momentum) * self._bias_offset + momentum * desired

    def apply_covariate_refresh(
        self,
        *,
        features: np.ndarray,
        reference_confidence: float,
        observed_confidence: float,
        intensity: int = 2,
    ) -> None:
        passes = max(1, intensity + 1)
        self.refresh_batch_norm(features, passes=passes)
        self.recalibrate_temperature(
            reference_confidence=reference_confidence,
            observed_confidence=observed_confidence,
            momentum=min(0.28, 0.10 + 0.05 * intensity),
        )
        self._feature_correction_strength = min(
            0.90,
            self._feature_correction_strength + 0.03 * max(1, intensity),
        )

    def apply_latent_recenter(self, features: np.ndarray, *, momentum: float = 0.12) -> None:
        array = np.asarray(features, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        batch_mean = array.mean(axis=0)
        if self._running_feature_mean is not None:
            self._running_feature_mean = (1.0 - momentum) * self._running_feature_mean + momentum * batch_mean
        self._feature_correction_strength = min(0.75, self._feature_correction_strength + momentum)

    def adapt(
        self,
        features: np.ndarray,
        probabilities: list[float],
        *,
        learning_rate: float,
        confidence_threshold: float,
        anchor_strength: float,
        entropy_weight: float,
        max_parameter_drift: float,
        steps: int = 2,
    ) -> float:
        batch = np.asarray(features, dtype=np.float32)
        probabilities_array = np.asarray(probabilities, dtype=np.float32)
        confident_mask = (probabilities_array >= confidence_threshold) | (
            probabilities_array <= 1.0 - confidence_threshold
        )
        selected = int(confident_mask.sum())
        if selected == 0:
            return 0.0
        selected_fraction = selected / max(1, len(probabilities_array))
        confident_features = batch[confident_mask]
        confident_probs = probabilities_array[confident_mask]
        pseudo_labels = (confident_probs >= 0.5).astype(np.float32)
        self.refresh_batch_norm(confident_features, passes=max(1, steps))
        self.apply_label_shift_correction(
            source_positive_rate=self._source_positive_rate,
            target_positive_rate=float(pseudo_labels.mean()),
            momentum=min(0.65, 0.20 + learning_rate * 6.0),
        )
        observed_confidence = float(np.maximum(confident_probs, 1.0 - confident_probs).mean())
        self.recalibrate_temperature(
            reference_confidence=max(0.62, 1.0 - anchor_strength),
            observed_confidence=observed_confidence,
            momentum=min(0.5, 0.15 + entropy_weight),
        )
        max_strength = min(0.9, max_parameter_drift)
        self._feature_correction_strength = min(
            max_strength,
            self._feature_correction_strength + 0.10 + 0.20 * selected_fraction,
        )
        return float(selected_fraction)

    def _transform_features(self, features: np.ndarray) -> np.ndarray:
        if (
            self._source_feature_mean is None
            or self._source_feature_std is None
            or self._running_feature_mean is None
            or self._running_feature_std is None
            or self._feature_correction_strength <= 0.0
        ):
            return features
        centered = (features - self._running_feature_mean) / np.clip(self._running_feature_std, 1e-3, None)
        corrected = centered * self._source_feature_std + self._source_feature_mean
        alpha = float(np.clip(self._feature_correction_strength, 0.0, 1.0))
        return ((1.0 - alpha) * features + alpha * corrected).astype(np.float32)


class BlackBoxModelAdapter:
    """Monitor-only adapter for hosted models that expose predictions but not weights."""

    adapter_kind = "black_box"

    def __init__(
        self,
        predict_proba_fn: Any,
        *,
        model_version: str = "external-v1",
    ) -> None:
        self._predict_proba_fn = predict_proba_fn
        self.model_version = model_version
        self.supports_adaptation = False

    def predict_proba(self, features: np.ndarray) -> list[float]:
        output = self._predict_proba_fn(features)
        if isinstance(output, np.ndarray):
            if output.ndim == 2:
                return output[:, 1].tolist() if output.shape[1] > 1 else output[:, 0].tolist()
            return output.tolist()
        return list(output)

    def predict(self, features: np.ndarray) -> list[int]:
        return [1 if probability >= 0.5 else 0 for probability in self.predict_proba(features)]

    def export_snapshot(self) -> AdapterSnapshot:
        return AdapterSnapshot(adapter_kind=self.adapter_kind, payload=None, version=self.model_version)

    def load_snapshot(self, snapshot: AdapterSnapshot) -> None:
        del snapshot

    def reset(self) -> None:
        return None

    def parameter_drift(self) -> float:
        return 0.0

    def refresh_batch_norm(self, features: np.ndarray, passes: int = 1) -> None:
        del features, passes

    def recalibrate_temperature(
        self,
        *,
        reference_confidence: float,
        observed_confidence: float,
        momentum: float = 0.25,
    ) -> None:
        del reference_confidence, observed_confidence, momentum

    def apply_label_shift_correction(
        self,
        *,
        source_positive_rate: float,
        target_positive_rate: float,
        momentum: float = 0.35,
    ) -> None:
        del source_positive_rate, target_positive_rate, momentum

    def apply_covariate_refresh(
        self,
        *,
        features: np.ndarray,
        reference_confidence: float,
        observed_confidence: float,
        intensity: int = 2,
    ) -> None:
        del features, reference_confidence, observed_confidence, intensity

    def apply_latent_recenter(self, features: np.ndarray, *, momentum: float = 0.12) -> None:
        del features, momentum

    def adapt(
        self,
        features: np.ndarray,
        probabilities: list[float],
        *,
        learning_rate: float,
        confidence_threshold: float,
        anchor_strength: float,
        entropy_weight: float,
        max_parameter_drift: float,
        steps: int = 2,
    ) -> float:
        del (
            features,
            probabilities,
            learning_rate,
            confidence_threshold,
            anchor_strength,
            entropy_weight,
            max_parameter_drift,
            steps,
        )
        return 0.0


def save_torch_adapter_checkpoint(adapter: TorchTabularModelAdapter, path: str | Path) -> None:
    import torch

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_version": adapter.model_version,
            "state_dict": adapter.inner.network.state_dict(),
            "temperature": adapter.inner.temperature,
            "bias_offset": adapter.inner.bias_offset,
        },
        checkpoint_path,
    )


def load_torch_adapter_checkpoint(
    *,
    input_dim: int,
    path: str | Path,
    model_version: str | None = None,
) -> TorchTabularModelAdapter:
    import torch

    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = TorchTabularAdapterModel(input_dim)
    model.network.load_state_dict(checkpoint["state_dict"])
    model.temperature = float(checkpoint.get("temperature", 1.0))
    model.bias_offset = float(checkpoint.get("bias_offset", 0.0))
    model._source_state = {key: value.detach().clone() for key, value in model.network.state_dict().items()}
    model._source_adaptation_state = model._capture_adaptation_state()
    version = model_version or str(checkpoint.get("model_version", "source-v1"))
    return TorchTabularModelAdapter(model, model_version=version)


def clone_model_adapter(adapter: ModelAdapter) -> ModelAdapter:
    """Create a fresh adapter instance with the same model state."""

    if isinstance(adapter, TorchTabularModelAdapter):
        return TorchTabularModelAdapter(adapter.inner.clone(), model_version=adapter.model_version)
    if isinstance(adapter, SklearnModelAdapter):
        snapshot = adapter.export_snapshot()
        estimator = pickle.loads(snapshot.payload["estimator"])
        clone = SklearnModelAdapter(estimator, model_version=adapter.model_version)
        clone.load_snapshot(snapshot)
        return clone
    if isinstance(adapter, BlackBoxModelAdapter):
        return BlackBoxModelAdapter(adapter._predict_proba_fn, model_version=adapter.model_version)
    raise TypeError(f"unsupported adapter type for cloning: {type(adapter)!r}")
