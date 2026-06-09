from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime.config import RuntimeConfig
from ..runtime.layer import ReliabilityLayer, build_reliability_layer_from_reference_batches
from ..runtime.model_adapter import SklearnModelAdapter, TorchTabularModelAdapter, load_torch_adapter_checkpoint
from ..tabular_benchmark import TabularBatch
from .config import ServingConfig


def _build_audit_sinks(serving: ServingConfig) -> list:
    """Build centralized audit sinks from serving config."""
    from ..runtime.audit import JsonlFileSink, KafkaAuditSink
    sinks = []
    if serving.audit_kafka_bootstrap and serving.audit_kafka_topic:
        sinks.append(KafkaAuditSink(
            bootstrap_servers=serving.audit_kafka_bootstrap,
            topic=serving.audit_kafka_topic,
        ))
    if serving.audit_jsonl_sink_path:
        sinks.append(JsonlFileSink(serving.audit_jsonl_sink_path))
    return sinks


def build_layer_for_serving(
    runtime_config: RuntimeConfig,
    serving: ServingConfig,
) -> ReliabilityLayer:
    audit_sinks = _build_audit_sinks(serving)
    if serving.model_bundle:
        layer = _build_from_bundle(runtime_config, serving.model_bundle)
    elif serving.adapter_kind == "sklearn":
        layer = _build_sklearn_layer(runtime_config, serving)
    elif serving.adapter_kind == "torch_tabular":
        layer = _build_torch_layer(runtime_config, serving)
    else:
        layer = _build_demo_tabular_layer(runtime_config)
    if audit_sinks:
        layer._governance._sinks = audit_sinks  # type: ignore[attr-defined]
    return layer


def _build_from_bundle(runtime_config: RuntimeConfig, bundle_name: str) -> ReliabilityLayer:
    from ..replay.real_data import REAL_DATA_LOADERS

    loader = REAL_DATA_LOADERS.get(bundle_name)
    if loader is None:
        raise ValueError(f"unknown model_bundle {bundle_name!r}; available: {sorted(REAL_DATA_LOADERS)}")
    bundle = loader(
        steps=runtime_config.replay.max_steps or 24,
        batch_size=runtime_config.replay.batch_size,
    )
    layer: ReliabilityLayer = bundle.build_layer(runtime_config)
    layer._expected_feature_dim = bundle.feature_dim  # type: ignore[attr-defined]
    return layer


def _build_sklearn_layer(runtime_config: RuntimeConfig, serving: ServingConfig) -> ReliabilityLayer:
    if not serving.sklearn_model_path:
        raise ValueError("serving.sklearn_model_path is required for adapter_kind=sklearn")
    path = Path(serving.sklearn_model_path)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import joblib

        estimator = joblib.load(path)
    except ImportError:
        with path.open("rb") as handle:
            estimator = pickle.load(handle)

    adapter = SklearnModelAdapter(estimator, model_version=runtime_config.model_version)
    reference_batches = _load_reference_batches(serving.reference_batches_path, runtime_config.replay.batch_size)
    layer = build_reliability_layer_from_reference_batches(adapter, reference_batches, config=runtime_config)
    layer._expected_feature_dim = _infer_feature_dim(adapter, serving.feature_dim)  # type: ignore[attr-defined]
    return layer


def _build_torch_layer(runtime_config: RuntimeConfig, serving: ServingConfig) -> ReliabilityLayer:
    if not serving.torch_checkpoint_path:
        raise ValueError("serving.torch_checkpoint_path is required for adapter_kind=torch_tabular")
    if serving.feature_dim is None:
        raise ValueError("serving.feature_dim is required for adapter_kind=torch_tabular")
    adapter = load_torch_adapter_checkpoint(
        input_dim=int(serving.feature_dim),
        path=serving.torch_checkpoint_path,
        model_version=runtime_config.model_version,
    )
    reference_batches = _load_reference_batches(serving.reference_batches_path, runtime_config.replay.batch_size)
    layer = build_reliability_layer_from_reference_batches(adapter, reference_batches, config=runtime_config)
    layer._expected_feature_dim = int(serving.feature_dim)  # type: ignore[attr-defined]
    return layer


def _build_demo_tabular_layer(runtime_config: RuntimeConfig) -> ReliabilityLayer:
    from ..tabular_benchmark import _build_real_tabular_source, _build_reference_batches
    from ..runtime.model_adapter import TorchTabularModelAdapter
    from ..torch_model import TorchTabularAdapterModel

    x_train, y_train, x_validation, y_validation, _, _ = _build_real_tabular_source(seed=7)
    model = TorchTabularAdapterModel(x_train.shape[1], seed=7)
    model.fit_source(x_train, y_train, x_validation, y_validation, epochs=10)
    adapter = TorchTabularModelAdapter(model, model_version=runtime_config.model_version)
    reference_batches = _build_reference_batches(
        x_validation,
        y_validation,
        batch_size=runtime_config.replay.batch_size,
        seed=7,
    )
    layer = build_reliability_layer_from_reference_batches(adapter, reference_batches, config=runtime_config)
    layer._expected_feature_dim = x_train.shape[1]  # type: ignore[attr-defined]
    return layer


def _load_reference_batches(path: str | None, batch_size: int) -> list[TabularBatch]:
    if path is None:
        from ..tabular_benchmark import _build_real_tabular_source, _build_reference_batches

        _, _, x_validation, y_validation, _, _ = _build_real_tabular_source(seed=7)
        return _build_reference_batches(
            x_validation,
            y_validation,
            batch_size=batch_size,
            seed=7,
        )
    
    # Try JSON format first (safer than pickle)
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, list):
            batches = []
            for batch_item in data:
                batch = TabularBatch(
                    features=np.array(batch_item.get("features"), dtype=np.float32),
                    labels=np.array(batch_item.get("labels"), dtype=np.int64),
                    regime=batch_item.get("regime", "live"),
                )
                batches.append(batch)
            return batches
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        pass  # Fall through to pickle for backward compatibility
    
    # Fall back to pickle for backward compatibility with existing saved batches
    # WARNING: Only use pickle with trusted data sources
    try:
        import logging
        logging.warning(
            f"Loading reference batches from pickle format (legacy). "
            f"Consider converting to JSON format for better security. "
            f"Path: {path}"
        )
        payload = pickle.loads(Path(path).read_bytes())
        if isinstance(payload, list):
            return payload
    except Exception as exc:
        raise ValueError(
            f"Failed to load reference_batches from {path}. "
            f"Must be a JSON list of batches or pickled list[TabularBatch]. "
            f"Error: {exc}"
        ) from exc
    
    raise ValueError(f"reference_batches_path must contain a list, got {type(payload)!r}")


def _infer_feature_dim(adapter: Any, configured: int | None) -> int:
    if configured is not None:
        return int(configured)
    if hasattr(adapter, "_estimator") and hasattr(adapter._estimator, "coef_"):
        return int(adapter._estimator.coef_.shape[1])
    raise ValueError("could not infer feature_dim; set serving.feature_dim in config")


def expected_feature_dim(layer: ReliabilityLayer) -> int | None:
    return getattr(layer, "_expected_feature_dim", None)
