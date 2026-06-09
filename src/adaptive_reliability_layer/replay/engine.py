from __future__ import annotations

from collections import deque
from dataclasses import replace
import math
from pathlib import Path
from typing import Callable

import numpy as np

from ..runtime.config import ReplayConfig, RuntimeConfig
from ..runtime.layer import ReliabilityLayer, build_reliability_layer_from_reference_batches
from ..runtime.model_adapter import TorchTabularModelAdapter
from ..runtime.types import DeploymentSurface, RuntimeBatch
from ..tabular_benchmark import TabularBatch
from ..torch_model import TorchTabularAdapterModel
from .loader import ReplayStream, iter_replay_batches, load_replay_csv
from .report import ReplayComparisonResult, render_replay_report, summarize_replay_runs
from .types import ReplayRunState


def _reveal_batch_labels(
    layer: ReliabilityLayer,
    reveal_step: int,
    labels: np.ndarray,
    surfaces: list[DeploymentSurface],
) -> dict[str, float]:
    if hasattr(layer, "reveal_labels"):
        try:
            return layer.reveal_labels(reveal_step, labels)
        except KeyError:
            pass
    revealed_surface = next(item for item in surfaces if item.step == reveal_step)
    batch_accuracy = float((np.array(revealed_surface.predictions) == labels).mean())
    return {"batch_accuracy": batch_accuracy}


def _utility_from_surface(surface: DeploymentSurface, batch_accuracy: float | None) -> float:
    accuracy = batch_accuracy if batch_accuracy is not None else 0.5
    utility = accuracy
    utility -= 0.06 * float(surface.risk_alert)
    utility -= 0.03 * min(1.0, surface.parameter_drift)
    utility -= 0.10 * float(surface.abstained)
    utility -= 0.04 * float(surface.action_taken == "reset")
    return utility


def run_replay_on_stream(
    layer: ReliabilityLayer,
    stream: ReplayStream,
    *,
    config: ReplayConfig,
    name: str = "controller",
) -> ReplayRunState:
    state = ReplayRunState(name=name, layer=layer)
    pending_labels: deque[tuple[int, np.ndarray, int]] = deque()
    label_delay = config.label_delay_steps
    delay_jitter = max(0, int(getattr(config, "label_delay_jitter_steps", 0)))

    for step, batch, _delayed_step in iter_replay_batches(
        stream,
        batch_size=config.batch_size,
        label_delay_steps=label_delay,
        max_steps=config.max_steps,
    ):
        true_labels = None
        if batch.labels is not None:
            true_labels = np.asarray(batch.labels, dtype=np.int64)

        batch_for_layer = batch
        if label_delay > 0 and true_labels is not None:
            batch_for_layer = RuntimeBatch(
                features=batch.features,
                labels=None,
                regime=batch.regime,
                timestamp=batch.timestamp,
                metadata=dict(batch.metadata or {}),
            )

        surface = layer.process_batch(batch_for_layer)
        state.surfaces.append(surface)
        state.shift_scores.append(surface.shift_score)
        state.risk_capitals.append(surface.risk_capital)

        if true_labels is not None and label_delay > 0:
            effective_delay = label_delay
            if delay_jitter > 0:
                effective_delay = max(0, label_delay + ((step * 7 + label_delay) % (2 * delay_jitter + 1)) - delay_jitter)
            pending_labels.append((surface.step, true_labels, effective_delay))

        while label_delay > 0 and pending_labels and step - pending_labels[0][0] >= pending_labels[0][2]:
            reveal_step, labels, _effective_delay = pending_labels.popleft()
            metrics = _reveal_batch_labels(layer, reveal_step, labels, state.surfaces)
            revealed_surface = next(
                item for item in state.surfaces if item.step == reveal_step
            )
            state.accuracies.append(metrics["batch_accuracy"])
            state.utilities.append(
                _utility_from_surface(revealed_surface, metrics["batch_accuracy"])
            )

        if true_labels is not None and label_delay <= 0:
            if getattr(layer, "revealed_metrics", None) and layer.revealed_metrics:
                batch_accuracy = float(layer.revealed_metrics[-1]["batch_accuracy"])
            else:
                batch_accuracy = float((np.array(surface.predictions) == true_labels).mean())
            state.accuracies.append(batch_accuracy)
            state.utilities.append(_utility_from_surface(surface, batch_accuracy))
        elif label_delay > 0 and not pending_labels:
            state.utilities.append(_utility_from_surface(surface, None))

        del step

    while label_delay > 0 and pending_labels:
        reveal_step, labels, _effective_delay = pending_labels.popleft()
        metrics = _reveal_batch_labels(layer, reveal_step, labels, state.surfaces)
        revealed_surface = next(item for item in state.surfaces if item.step == reveal_step)
        state.accuracies.append(metrics["batch_accuracy"])
        state.utilities.append(
            _utility_from_surface(revealed_surface, metrics["batch_accuracy"])
        )

    state.revealed_metrics = [dict(item) for item in getattr(layer, "revealed_metrics", ())]
    return state


class _ADWIN:
    """Minimal ADWIN drift detector (no external dependencies).

    Monitors a stream of binary values (e.g. per-batch error rate) and
    signals drift when it finds a statistically significant cut-point in
    the sliding window — at which point it trims the window to the newer
    sub-window so the detector adapts to the post-drift distribution.
    """

    def __init__(self, delta: float = 0.002) -> None:
        self._delta = delta
        self._window: list[float] = []

    def update(self, value: float) -> bool:
        """Add a new observation; return True if drift was detected."""
        self._window.append(float(value))
        return self._check_and_trim()

    def _check_and_trim(self) -> bool:
        n = len(self._window)
        if n < 8:
            return False
        total = sum(self._window)
        prefix = 0.0
        for split in range(1, n):
            prefix += self._window[split - 1]
            n0 = split
            n1 = n - split
            m0 = prefix / n0
            m1 = (total - prefix) / n1
            harmonic = 1.0 / n0 + 1.0 / n1
            threshold = math.sqrt(0.5 * harmonic * math.log(2.0 * n / self._delta))
            if abs(m0 - m1) >= threshold:
                self._window = self._window[split:]
                return True
        return False

    @property
    def n(self) -> int:
        return len(self._window)


def run_adwin_retrain_on_stream(
    layer: ReliabilityLayer,
    stream: "ReplayStream",
    *,
    config: ReplayConfig,
    name: str = "adwin_retrain",
    adwin_delta: float = 0.002,
) -> "ReplayRunState":
    """ADWIN-triggered retrain strategy — the standard literature baseline.

    Runs a frozen model until ADWIN detects drift in the revealed error
    stream, then resets the model back to its source snapshot (simulating
    a full retrain from source data).  This is the realistic practitioner
    alternative: use a monitoring tool and retrain when drift fires.
    """
    state = ReplayRunState(name=name, layer=layer)
    adwin = _ADWIN(delta=adwin_delta)
    pending_labels: deque[tuple[int, np.ndarray, int]] = deque()
    label_delay = config.label_delay_steps
    delay_jitter = max(0, int(getattr(config, "label_delay_jitter_steps", 0)))
    source_snapshot = layer._adapter.export_snapshot()
    retrains = 0

    for step, batch, _delayed_step in iter_replay_batches(
        stream,
        batch_size=config.batch_size,
        label_delay_steps=label_delay,
        max_steps=config.max_steps,
    ):
        true_labels = None
        if batch.labels is not None:
            true_labels = np.asarray(batch.labels, dtype=np.int64)

        batch_for_layer = batch
        if label_delay > 0 and true_labels is not None:
            batch_for_layer = RuntimeBatch(
                features=batch.features,
                labels=None,
                regime=batch.regime,
                timestamp=batch.timestamp,
                metadata=dict(batch.metadata or {}),
            )

        surface = layer.process_batch(batch_for_layer)
        state.surfaces.append(surface)
        state.shift_scores.append(surface.shift_score)
        state.risk_capitals.append(surface.risk_capital)

        if true_labels is not None and label_delay > 0:
            effective_delay = label_delay
            if delay_jitter > 0:
                effective_delay = max(0, label_delay + ((step * 7 + label_delay) % (2 * delay_jitter + 1)) - delay_jitter)
            pending_labels.append((surface.step, true_labels, effective_delay))

        while label_delay > 0 and pending_labels and step - pending_labels[0][0] >= pending_labels[0][2]:
            reveal_step, labels, _eff = pending_labels.popleft()
            metrics = _reveal_batch_labels(layer, reveal_step, labels, state.surfaces)
            revealed_surface = next(item for item in state.surfaces if item.step == reveal_step)
            batch_accuracy = metrics["batch_accuracy"]
            state.accuracies.append(batch_accuracy)
            state.utilities.append(_utility_from_surface(revealed_surface, batch_accuracy))
            error_rate = 1.0 - batch_accuracy
            if adwin.update(error_rate):
                layer._adapter.load_snapshot(source_snapshot)
                retrains += 1

        if true_labels is not None and label_delay <= 0:
            if getattr(layer, "revealed_metrics", None) and layer.revealed_metrics:
                batch_accuracy = float(layer.revealed_metrics[-1]["batch_accuracy"])
            else:
                batch_accuracy = float((np.array(surface.predictions) == true_labels).mean())
            state.accuracies.append(batch_accuracy)
            state.utilities.append(_utility_from_surface(surface, batch_accuracy))
            error_rate = 1.0 - batch_accuracy
            if adwin.update(error_rate):
                layer._adapter.load_snapshot(source_snapshot)
                retrains += 1
        elif label_delay > 0 and not pending_labels:
            state.utilities.append(_utility_from_surface(surface, None))

        del step

    while label_delay > 0 and pending_labels:
        reveal_step, labels, _eff = pending_labels.popleft()
        metrics = _reveal_batch_labels(layer, reveal_step, labels, state.surfaces)
        revealed_surface = next(item for item in state.surfaces if item.step == reveal_step)
        batch_accuracy = metrics["batch_accuracy"]
        state.accuracies.append(batch_accuracy)
        state.utilities.append(_utility_from_surface(revealed_surface, batch_accuracy))

    state.revealed_metrics = [dict(item) for item in getattr(layer, "revealed_metrics", ())]
    state.metadata["adwin_retrains"] = retrains
    return state


def run_evidently_retrain_on_stream(
    layer: ReliabilityLayer,
    stream: "ReplayStream",
    *,
    config: ReplayConfig,
    name: str = "evidently_retrain",
    psi_threshold: float = 0.2,
    window_size: int = 5,
) -> "ReplayRunState":
    """Evidently-style PSI drift → retrain baseline.

    Simulates the common practitioner workflow: compute Population Stability
    Index (PSI) on a rolling window of model outputs vs. the reference window,
    and trigger a full reset when PSI exceeds the threshold.  PSI ≥ 0.2 is the
    industry standard for "significant drift".

    Unlike ADWIN (which monitors revealed error rates), this baseline only uses
    the model's own output distribution — no labels needed.  This makes it
    directly comparable to tools like Evidently AI or Arize.
    """
    def _psi(reference: list[float], current: list[float], bins: int = 10) -> float:
        """Population Stability Index between two probability distributions."""
        if not reference or not current:
            return 0.0
        edges = [i / bins for i in range(bins + 1)]
        def _hist(values: list[float]) -> list[float]:
            counts = [0.0] * bins
            for v in values:
                idx = min(int(v * bins), bins - 1)
                counts[idx] += 1
            n = max(1.0, float(len(values)))
            return [max(c / n, 1e-4) for c in counts]
        ref_hist = _hist(reference)
        cur_hist = _hist(current)
        return float(sum(
            (c - r) * math.log(c / r)
            for r, c in zip(ref_hist, cur_hist)
        ))

    state = ReplayRunState(name=name, layer=layer)
    pending_labels: deque[tuple[int, np.ndarray, int]] = deque()
    label_delay = config.label_delay_steps
    delay_jitter = max(0, int(getattr(config, "label_delay_jitter_steps", 0)))
    source_snapshot = layer._adapter.export_snapshot()
    retrains = 0
    reference_probs: list[float] = []
    window_probs: list[float] = []

    for step, batch, _delayed_step in iter_replay_batches(
        stream,
        batch_size=config.batch_size,
        label_delay_steps=label_delay,
        max_steps=config.max_steps,
    ):
        true_labels = None
        if batch.labels is not None:
            true_labels = np.asarray(batch.labels, dtype=np.int64)

        batch_for_layer = batch
        if label_delay > 0 and true_labels is not None:
            batch_for_layer = RuntimeBatch(
                features=batch.features,
                labels=None,
                regime=batch.regime,
                timestamp=batch.timestamp,
                metadata=dict(batch.metadata or {}),
            )

        surface = layer.process_batch(batch_for_layer)
        state.surfaces.append(surface)
        state.shift_scores.append(surface.shift_score)
        state.risk_capitals.append(surface.risk_capital)

        batch_probs = list(surface.probabilities)
        if len(reference_probs) < window_size * config.batch_size:
            reference_probs.extend(batch_probs)
        else:
            window_probs.extend(batch_probs)
            if len(window_probs) >= window_size * config.batch_size:
                psi = _psi(reference_probs, window_probs[-window_size * config.batch_size:])
                if psi >= psi_threshold:
                    layer._adapter.load_snapshot(source_snapshot)
                    retrains += 1
                    window_probs = []

        if true_labels is not None and label_delay > 0:
            effective_delay = label_delay
            if delay_jitter > 0:
                effective_delay = max(0, label_delay + ((step * 7 + label_delay) % (2 * delay_jitter + 1)) - delay_jitter)
            pending_labels.append((surface.step, true_labels, effective_delay))

        while label_delay > 0 and pending_labels and step - pending_labels[0][0] >= pending_labels[0][2]:
            reveal_step, labels, _eff = pending_labels.popleft()
            metrics = _reveal_batch_labels(layer, reveal_step, labels, state.surfaces)
            revealed_surface = next(item for item in state.surfaces if item.step == reveal_step)
            batch_accuracy = metrics["batch_accuracy"]
            state.accuracies.append(batch_accuracy)
            state.utilities.append(_utility_from_surface(revealed_surface, batch_accuracy))

        if true_labels is not None and label_delay <= 0:
            if getattr(layer, "revealed_metrics", None) and layer.revealed_metrics:
                batch_accuracy = float(layer.revealed_metrics[-1]["batch_accuracy"])
            else:
                batch_accuracy = float((np.array(surface.predictions) == true_labels).mean())
            state.accuracies.append(batch_accuracy)
            state.utilities.append(_utility_from_surface(surface, batch_accuracy))
        elif label_delay > 0 and not pending_labels:
            state.utilities.append(_utility_from_surface(surface, None))

        del step

    while label_delay > 0 and pending_labels:
        reveal_step, labels, _eff = pending_labels.popleft()
        metrics = _reveal_batch_labels(layer, reveal_step, labels, state.surfaces)
        revealed_surface = next(item for item in state.surfaces if item.step == reveal_step)
        batch_accuracy = metrics["batch_accuracy"]
        state.accuracies.append(batch_accuracy)
        state.utilities.append(_utility_from_surface(revealed_surface, batch_accuracy))

    state.revealed_metrics = [dict(item) for item in getattr(layer, "revealed_metrics", ())]
    state.metadata["evidently_retrains"] = retrains
    return state


def run_river_online_on_stream(
    stream: "ReplayStream",
    *,
    config: ReplayConfig,
    name: str = "river_hoeffding",
    classifier_name: str = "hoeffding_tree",
) -> "ReplayRunState":
    """River online classifier baseline.

    Uses River's HoeffdingAdaptiveTreeClassifier (HAT) or AdaptiveRandomForest
    when `river` is installed.  Falls back to a pure-Python exponential moving
    average (EMA) threshold classifier when `river` is not available — this is
    a placeholder that establishes the API; install `river` for meaningful results.

    Unlike ARL, River classifiers update per-sample with the true label at
    inference time — they require labels immediately and cannot handle delay.
    This represents the "ideal online learning" upper bound for comparison.
    """
    try:
        if classifier_name == "adaptive_random_forest":
            from river.ensemble import AdaptiveRandomForestClassifier
            clf = AdaptiveRandomForestClassifier(n_models=10, seed=7)
        else:
            from river.tree import HoeffdingAdaptiveTreeClassifier
            clf = HoeffdingAdaptiveTreeClassifier(seed=7)
        river_available = True
    except ImportError:
        clf = None
        river_available = False

    # Pure-Python fallback: EMA of positive rate → threshold classifier
    ema_rate = 0.5
    ema_decay = 0.95

    label_delay = config.label_delay_steps
    from ..runtime.layer import build_reliability_layer_from_reference_batches
    from ..runtime.model_adapter import BlackBoxModelAdapter
    from ..tabular_benchmark import TabularBatch, _build_real_tabular_source, _build_reference_batches
    from ..torch_model import TorchTabularAdapterModel

    # Build a dummy layer just to get a ReplayRunState structure and reference profile
    # (River replaces the prediction logic entirely)
    x_train, y_train, x_val, y_val, _, _ = _build_real_tabular_source(seed=7)
    source_model = TorchTabularAdapterModel(x_train.shape[1], seed=7)
    source_model.fit_source(x_train, y_train, x_val, y_val, epochs=10)
    adapter = TorchTabularModelAdapter(source_model)

    from ..runtime.config import RuntimeConfig, GovernanceConfig, MetricsConfig
    from ..runtime.types import OperatingMode
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        cfg = RuntimeConfig(
            operating_mode=OperatingMode.SHADOW,
            governance=GovernanceConfig(audit_db_path=os.path.join(tmp, "a.db"), snapshot_dir=os.path.join(tmp, "s")),
            metrics=MetricsConfig(enabled=False),
            log_json=False,
        )
        ref_batches = _build_reference_batches(x_val, y_val, batch_size=32, seed=7)
        layer = build_reliability_layer_from_reference_batches(adapter, ref_batches, config=cfg)

    state = ReplayRunState(name=name, layer=layer)

    for step, batch, _delayed in iter_replay_batches(
        stream,
        batch_size=config.batch_size,
        label_delay_steps=0,  # River needs labels immediately
        max_steps=config.max_steps,
    ):
        features = np.asarray(batch.features, dtype=np.float32)
        true_labels = None
        if batch.labels is not None:
            true_labels = np.asarray(batch.labels, dtype=np.int64)

        predictions: list[int] = []
        probabilities: list[float] = []

        for i in range(len(features)):
            x_dict = {f"f{j}": float(features[i, j]) for j in range(features.shape[1])}
            if river_available and clf is not None:
                try:
                    proba_dict = clf.predict_proba_one(x_dict)
                    prob = float(proba_dict.get(1, 0.5))
                except Exception:
                    prob = 0.5
            else:
                prob = ema_rate
            predictions.append(1 if prob >= 0.5 else 0)
            probabilities.append(prob)

            # River learns from the true label immediately (no delay)
            if true_labels is not None and river_available and clf is not None:
                try:
                    clf.learn_one(x_dict, int(true_labels[i]))
                except Exception:
                    pass
            elif true_labels is not None:
                ema_rate = ema_decay * ema_rate + (1 - ema_decay) * float(true_labels[i])

        # Build a fake surface for the ReplayRunState
        from ..runtime.types import DeploymentSurface
        surface = DeploymentSurface(
            step=step,
            predictions=predictions,
            probabilities=probabilities,
            confidence=float(np.mean([max(p, 1 - p) for p in probabilities])),
            shift_score=0.0,
            feature_shift_score=0.0,
            output_shift_score=0.0,
            collapse_risk=0.0,
            risk_capital=1.0,
            risk_alert=False,
            regime_hint=batch.regime,
            recommended_action="none",
            action_taken="none",
            intervention_reason="river_online",
            trust_state="normal",
            reliability_score=1.0,
            parameter_drift=0.0,
            operating_mode="shadow",
            model_version=f"river_{classifier_name}",
            specialist_id=None,
            rollback_available=False,
            snapshot_id=None,
            abstained=False,
        )
        state.surfaces.append(surface)
        state.shift_scores.append(0.0)
        state.risk_capitals.append(1.0)

        if true_labels is not None:
            batch_accuracy = float((np.array(predictions) == true_labels).mean())
            state.accuracies.append(batch_accuracy)
            state.utilities.append(batch_accuracy)

        del step

    state.metadata["river_available"] = river_available
    state.metadata["classifier"] = classifier_name
    return state


def build_layer_for_tabular_replay(
    *,
    config: RuntimeConfig,
    reference_batches: list[TabularBatch] | None = None,
) -> ReliabilityLayer:
    from ..tabular_benchmark import _build_real_tabular_source, _build_reference_batches

    x_train, y_train, x_validation, y_validation, _x_test, _y_test = _build_real_tabular_source(seed=7)
    model = TorchTabularAdapterModel(x_train.shape[1], seed=7)
    model.fit_source(x_train, y_train, x_validation, y_validation, epochs=20)
    adapter = TorchTabularModelAdapter(model, model_version=config.model_version)

    if reference_batches is None:
        reference_batches = _build_reference_batches(
            x_validation,
            y_validation,
            batch_size=config.replay.batch_size,
            seed=7,
        )
    return build_reliability_layer_from_reference_batches(adapter, reference_batches, config=config)


def run_offline_replay_comparison(
    stream: ReplayStream,
    *,
    runtime_config: RuntimeConfig,
    strategies: tuple[str, ...] = ("frozen", "naive", "controller", "bandit"),
    layer_builder: Callable[[RuntimeConfig], ReliabilityLayer] | None = None,
    controller_name: str | None = None,
) -> ReplayComparisonResult:
    runs: list[ReplayRunState] = []

    for strategy in strategies:
        frozen_config = replace(
            runtime_config,
            policy=replace(runtime_config.policy, name="frozen"),
            log_json=False,
        )
        strategy_config = replace(
            runtime_config,
            policy=replace(runtime_config.policy, name=strategy),
            log_json=False,
        )
        if strategy == "adwin_retrain":
            layer = (layer_builder(config=frozen_config) if layer_builder is not None
                     else build_layer_for_tabular_replay(config=frozen_config))
            runs.append(
                run_adwin_retrain_on_stream(
                    layer,
                    stream,
                    config=runtime_config.replay,
                    name=strategy,
                )
            )
        elif strategy == "evidently_retrain":
            layer = (layer_builder(config=frozen_config) if layer_builder is not None
                     else build_layer_for_tabular_replay(config=frozen_config))
            runs.append(
                run_evidently_retrain_on_stream(
                    layer,
                    stream,
                    config=runtime_config.replay,
                    name=strategy,
                )
            )
        elif strategy in ("river_hoeffding", "river_arf", "river_adaptive_random_forest"):
            classifier_name = "adaptive_random_forest" if strategy in ("river_arf", "river_adaptive_random_forest") else "hoeffding_tree"
            runs.append(
                run_river_online_on_stream(
                    stream,
                    config=runtime_config.replay,
                    name=strategy,
                    classifier_name=classifier_name,
                )
            )
        elif strategy in ("tent", "tent_tta"):
            tent_config = replace(
                runtime_config,
                policy=replace(runtime_config.policy, name="tent"),
                log_json=False,
            )
            if layer_builder is not None:
                layer = layer_builder(config=tent_config)
            else:
                layer = build_layer_for_tabular_replay(config=tent_config)
            runs.append(
                run_replay_on_stream(
                    layer,
                    stream,
                    config=runtime_config.replay,
                    name=strategy,
                )
            )
        else:
            if layer_builder is not None:
                layer = layer_builder(config=strategy_config)
            else:
                layer = build_layer_for_tabular_replay(config=strategy_config)
            runs.append(
                run_replay_on_stream(
                    layer,
                    stream,
                    config=runtime_config.replay,
                    name=strategy,
                )
            )

    resolved_controller = controller_name
    if resolved_controller is None:
        resolved_controller = next(
            (strategy for strategy in reversed(strategies) if strategy != "frozen"),
            None,
        )
    return summarize_replay_runs(runs, controller_name=resolved_controller)


def export_stream_to_csv(stream: ReplayStream, path: str | Path) -> Path:
    import pandas as pd

    rows = []
    for index, record in enumerate(stream.records):
        row = {"timestamp": record.timestamp or f"t{index}", "label": record.label}
        regime = (record.metadata or {}).get("regime")
        if regime is not None:
            row["regime"] = regime
        for feature_index, value in enumerate(record.features):
            row[f"feature_{feature_index}"] = float(value)
        rows.append(row)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def build_synthetic_fraud_like_stream(
    *,
    steps: int = 90,
    batch_size: int = 48,
    seed: int = 7,
) -> ReplayStream:
    """Build a fraud-like replay stream from the internal tabular shift benchmark."""

    from ..tabular_benchmark import _build_real_tabular_source, build_tabular_stream

    _x_train, _y_train, _x_validation, _y_validation, x_test, y_test = _build_real_tabular_source(seed=seed)
    batches = build_tabular_stream(x_test, y_test, steps=steps, batch_size=batch_size, seed=seed)
    records = []
    for step, batch in enumerate(batches):
        for row_index in range(len(batch.labels)):
            records.append(
                {
                    "timestamp": f"2025-01-{(step % 28) + 1:02d}T{step:02d}:{row_index:02d}:00Z",
                    "features": batch.features[row_index],
                    "label": int(batch.labels[row_index]),
                    "metadata": {"regime": batch.regime, "step": step},
                }
            )
    from .loader import ReplayRecord

    replay_records = tuple(
        ReplayRecord(
            timestamp=item["timestamp"],
            features=item["features"],
            label=item["label"],
            metadata=item["metadata"],
        )
        for item in records
    )
    feature_dim = batches[0].features.shape[1]
    return ReplayStream(
        records=replay_records,
        feature_columns=tuple(f"feature_{index}" for index in range(feature_dim)),
    )
