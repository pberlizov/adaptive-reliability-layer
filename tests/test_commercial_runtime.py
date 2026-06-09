from __future__ import annotations

from dataclasses import replace
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from adaptive_reliability_layer.replay.engine import (
    build_layer_for_tabular_replay,
    build_synthetic_fraud_like_stream,
    run_offline_replay_comparison,
    run_replay_on_stream,
)
from adaptive_reliability_layer.replay.loader import iter_replay_batches, load_replay_csv
from adaptive_reliability_layer.replay.pilot import DEFAULT_PILOT, run_pilot_case_study
from adaptive_reliability_layer.risk import MartingaleRiskMonitor, RiskState
from adaptive_reliability_layer.runtime.action_gating import apply_operating_mode
from adaptive_reliability_layer.runtime.audit import AuditStore, GovernanceService, SnapshotStore
from adaptive_reliability_layer.runtime.config import RuntimeConfig, load_runtime_config
from adaptive_reliability_layer.runtime.layer import ReliabilityLayer, build_reliability_layer_from_reference_batches
from adaptive_reliability_layer.runtime.model_adapter import (
    BlackBoxModelAdapter,
    SklearnModelAdapter,
    TorchTabularModelAdapter,
)
from adaptive_reliability_layer.runtime.policy_state import export_policy_state, load_policy_state
from adaptive_reliability_layer.runtime.types import DeploymentSurface, InterventionDecision, OperatingMode, RuntimeBatch
from adaptive_reliability_layer.tabular_benchmark import (
    FraudRankDelayedBanditTabularPolicy,
    RegimeAwareDelayedBanditTabularPolicy,
    TabularBatch,
    TabularDecision,
    TabularReferenceProfile,
    TabularShiftSignal,
    _build_reference_batches,
)
from adaptive_reliability_layer.torch_model import TorchTabularAdapterModel


@pytest.fixture
def temp_governance_dirs(tmp_path: Path):
    audit_db = tmp_path / "audit.db"
    snapshot_dir = tmp_path / "snapshots"
    return audit_db, snapshot_dir


@pytest.fixture
def runtime_config(temp_governance_dirs) -> RuntimeConfig:
    audit_db, snapshot_dir = temp_governance_dirs
    base = load_runtime_config("configs/default.yaml")
    return RuntimeConfig(
        operating_mode=OperatingMode.SHADOW,
        bounded_auto_actions=base.bounded_auto_actions,
        model_version="test-v1",
        monitor=base.monitor,
        policy=base.policy,
        governance=base.governance.__class__(
            audit_db_path=str(audit_db),
            snapshot_dir=str(snapshot_dir),
            max_snapshots=20,
            policy_version="test",
            environment="test",
        ),
        metrics=base.metrics.__class__(enabled=False, prometheus_port=9091, namespace="arl_test"),
        replay=base.replay.__class__(batch_size=16, max_steps=6, label_delay_steps=0),
        sota=base.sota,
        log_json=False,
    )


@pytest.fixture
def trained_layer(runtime_config: RuntimeConfig) -> ReliabilityLayer:
    return build_layer_for_tabular_replay(config=runtime_config)


def test_config_loads_default_yaml():
    config = load_runtime_config("configs/default.yaml")
    assert config.operating_mode == OperatingMode.SHADOW
    assert "bn_refresh" in config.bounded_auto_actions


def test_martingale_risk_monitor_resets():
    monitor = MartingaleRiskMonitor([0.1, 0.2, 0.3], alert_threshold=2.0)
    for _ in range(5):
        monitor.update(5.0)
    assert monitor.update(5.0).capital >= 1.0
    monitor.reset()
    assert monitor.update(0.2).capital == pytest.approx(1.0, rel=0.5)


def test_shadow_mode_does_not_mutate_model(trained_layer: ReliabilityLayer):
    batch = RuntimeBatch(features=np.random.randn(16, 30).astype(np.float32))
    drift_before = trained_layer._adapter.parameter_drift()
    surface = trained_layer.process_batch(batch)
    drift_after = trained_layer._adapter.parameter_drift()
    assert surface.action_taken == "none"
    assert drift_before == drift_after


def test_bounded_auto_allows_low_risk_actions(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.BOUNDED_AUTO)
    layer = build_layer_for_tabular_replay(config=config)
    batch = RuntimeBatch(features=np.random.randn(16, 30).astype(np.float32))
    surface = layer.process_batch(batch)
    assert surface.operating_mode == "bounded_auto"


def test_bounded_auto_budget_downgrades_to_recommend(runtime_config: RuntimeConfig):
    config = replace(
        runtime_config,
        operating_mode=OperatingMode.BOUNDED_AUTO,
        safety_budget=runtime_config.safety_budget.__class__(
            enabled=True,
            window_steps=8,
            max_auto_actions_per_window=0,
            max_resets_per_window=0,
            downgrade_to_recommend=True,
        ),
    )
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            return TabularDecision(action="label_shift", reason="force_label_shift")

    layer._policy = FixedPolicy()  # type: ignore[attr-defined]
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        regime="budget_test",
        metadata={"controller_profile": "fraud"},
    )
    surface = layer.process_batch(batch)
    assert surface.action_taken == "none"
    assert surface.budget_limited is True
    assert surface.effective_operating_mode == "recommend"
    assert surface.recommended_action_requires_approval is True


def test_fraud_profile_allows_label_shift_for_label_drift(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.BOUNDED_AUTO)
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            return TabularDecision(action="label_shift", reason="force_label_shift")

    layer._policy = FixedPolicy()  # type: ignore[attr-defined]
    layer._monitor.evaluate = lambda features, probabilities: TabularShiftSignal(  # type: ignore[method-assign]
        score=1.3,
        feature_score=0.05,
        output_score=0.45,
        collapse_risk=0.05,
        alert=True,
        severe=False,
        mean_entropy=0.4,
        mean_probability=0.75,
        positive_rate=0.7,
        mean_confidence=0.88,
    )
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        regime="fraud_profile_test",
        metadata={"controller_profile": "fraud"},
    )
    surface = layer.process_batch(batch)
    assert surface.shift_signature == "label_drift"
    assert surface.controller_profile == "fraud"
    assert surface.action_taken == "label_shift"


def test_sensor_profile_blocks_label_shift_for_mixed_drift(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.BOUNDED_AUTO)
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            return TabularDecision(action="label_shift", reason="force_label_shift")

    layer._policy = FixedPolicy()  # type: ignore[attr-defined]
    layer._monitor.evaluate = lambda features, probabilities: TabularShiftSignal(  # type: ignore[method-assign]
        score=1.3,
        feature_score=0.05,
        output_score=0.45,
        collapse_risk=0.05,
        alert=True,
        severe=False,
        mean_entropy=0.4,
        mean_probability=0.75,
        positive_rate=0.7,
        mean_confidence=0.88,
    )
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        regime="sensor_profile_test",
        metadata={"controller_profile": "sensor"},
    )
    surface = layer.process_batch(batch)
    assert surface.shift_signature == "mixed_drift"
    assert surface.controller_profile == "sensor"
    assert surface.action_taken == "none"
    assert "bounded_auto_blocked:label_shift" in surface.intervention_reason


def test_sensor_profile_saturation_stands_down(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.BOUNDED_AUTO)
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            return TabularDecision(action="bn_refresh", reason="force_bn_refresh")

    layer._policy = FixedPolicy()  # type: ignore[attr-defined]
    layer._monitor.evaluate = lambda features, probabilities: TabularShiftSignal(  # type: ignore[method-assign]
        score=250.0,
        feature_score=12.0,
        output_score=0.2,
        collapse_risk=0.05,
        alert=True,
        severe=True,
        mean_entropy=0.25,
        mean_probability=0.61,
        positive_rate=0.59,
        mean_confidence=0.92,
    )
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        regime="sensor_saturated",
        metadata={"controller_profile": "sensor"},
    )
    surface = layer.process_batch(batch)
    assert surface.monitor_saturated is True
    assert surface.adaptation_opportunity_score == 0.0
    assert surface.action_taken == "none"
    assert surface.retrain_recommended is True


def test_sensor_profile_emits_reference_break_for_extreme_feature_mismatch(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.BOUNDED_AUTO)
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            return TabularDecision(action="bn_refresh", reason="force_bn_refresh")

    layer._policy = FixedPolicy()  # type: ignore[attr-defined]
    layer._monitor.evaluate = lambda features, probabilities: TabularShiftSignal(  # type: ignore[method-assign]
        score=8.5,
        feature_score=5.2,
        output_score=0.18,
        collapse_risk=0.08,
        alert=True,
        severe=True,
        mean_entropy=0.26,
        mean_probability=0.61,
        positive_rate=0.58,
        mean_confidence=0.90,
    )
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        regime="sensor_reference_break",
        metadata={"controller_profile": "sensor"},
    )
    surface = layer.process_batch(batch)
    assert surface.shift_signature == "reference_break"
    assert surface.action_taken == "none"
    assert surface.retrain_recommended is True


def test_sensor_safe_profile_blocks_covariate_refresh(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.BOUNDED_AUTO)
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            return TabularDecision(action="covariate_refresh", reason="force_covariate_refresh")

    layer._policy = FixedPolicy()  # type: ignore[attr-defined]
    layer._monitor.evaluate = lambda features, probabilities: TabularShiftSignal(  # type: ignore[method-assign]
        score=1.8,
        feature_score=2.1,
        output_score=0.08,
        collapse_risk=0.03,
        alert=True,
        severe=False,
        mean_entropy=0.22,
        mean_probability=0.62,
        positive_rate=0.58,
        mean_confidence=0.91,
    )
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        regime="sensor_safe_profile_test",
        metadata={"controller_profile": "sensor_safe"},
    )
    surface = layer.process_batch(batch)
    assert surface.shift_signature == "covariate_drift"
    assert surface.controller_profile == "sensor_safe"
    assert surface.action_taken == "none"
    assert "bounded_auto_blocked:covariate_refresh" in surface.intervention_reason


def test_sensor_covariate_drift_remaps_bn_refresh_to_covariate_refresh(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.BOUNDED_AUTO)
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            return TabularDecision(action="bn_refresh", reason="force_bn_refresh")

    layer._policy = FixedPolicy()  # type: ignore[attr-defined]
    layer._monitor.evaluate = lambda features, probabilities: TabularShiftSignal(  # type: ignore[method-assign]
        score=1.6,
        feature_score=1.9,
        output_score=0.08,
        collapse_risk=0.02,
        alert=True,
        severe=False,
        mean_entropy=0.21,
        mean_probability=0.60,
        positive_rate=0.57,
        mean_confidence=0.90,
    )
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        regime="sensor_covariate_refresh",
        metadata={"controller_profile": "sensor"},
    )
    surface = layer.process_batch(batch)
    assert surface.shift_signature == "covariate_drift"
    assert surface.controller_profile == "sensor"
    assert surface.recommended_action == "covariate_refresh"
    assert surface.action_taken == "covariate_refresh"


def test_sklearn_adapter_covariate_refresh_changes_probabilities():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    estimator = LogisticRegression(max_iter=200).fit(x_train, y_train)
    adapter = SklearnModelAdapter(
        estimator,
        source_feature_mean=x_train.mean(axis=0),
        source_feature_std=np.clip(x_train.std(axis=0), 1e-3, None),
        source_positive_rate=float(y_train.mean()),
    )
    shifted = (x_test[:32] * 1.25 + 0.18).astype(np.float32)
    before = np.asarray(adapter.predict_proba(shifted))
    adapter.apply_covariate_refresh(
        features=shifted,
        reference_confidence=0.82,
        observed_confidence=0.61,
        intensity=2,
    )
    after = np.asarray(adapter.predict_proba(shifted))
    assert not np.allclose(before, after)


def test_regime_aware_delayed_bandit_residual_correction_updates_probabilities():
    reference = TabularReferenceProfile(
        feature_mean=np.zeros(16, dtype=np.float32),
        feature_variance=np.ones(16, dtype=np.float32),
        mean_entropy=0.4,
        mean_probability=0.5,
        positive_rate=0.5,
        mean_confidence=0.8,
    )
    policy = RegimeAwareDelayedBanditTabularPolicy(reference, allowed_actions=("none",))
    batch = TabularBatch(
        features=np.random.randn(8, 16).astype(np.float32),
        labels=np.ones(8, dtype=np.int64),
        regime="delayed_residual",
    )
    signal = TabularShiftSignal(
        score=0.4,
        feature_score=0.1,
        output_score=0.05,
        collapse_risk=0.0,
        alert=False,
        severe=False,
        mean_entropy=0.35,
        mean_probability=0.45,
        positive_rate=0.5,
        mean_confidence=0.82,
    )
    risk_state = RiskState(raw_score=0.0, p_value=1.0, e_value=1.0, capital=1.0, alert=False)

    class DummyModel:
        pass

    decision = policy.apply(DummyModel(), signal, risk_state, batch, [0.45] * len(batch.labels))  # type: ignore[arg-type]
    feedback_state = policy.capture_feedback_state()
    assert feedback_state is not None
    before = policy.correct_probabilities([0.45, 0.45], signal=signal, risk_state=risk_state, batch=batch)
    policy.observe_delayed_outcome(
        feedback_state=feedback_state,
        model=DummyModel(),  # type: ignore[arg-type]
        batch=batch,
        signal=signal,
        risk_state=risk_state,
        decision=decision,
        batch_accuracy=1.0,
        reliability=0.9,
        utility=0.9,
        retrospective_reward=0.9,
        revealed_accuracy=1.0,
        revealed_coverage=1.0,
        revealed_baseline_accuracy=0.5,
        pending_delay_steps=6,
        pending_outstanding_count=2,
        revealed_mean_residual=0.55,
        predicted_positive_rate=0.45,
        revealed_positive_rate=1.0,
    )
    after = policy.correct_probabilities([0.45, 0.45], signal=signal, risk_state=risk_state, batch=batch)
    assert after[0] > before[0]
    threshold = policy.decision_threshold(signal=signal, risk_state=risk_state, batch=batch)
    assert threshold < 0.5
    diagnostics = policy.get_diagnostics()
    assert diagnostics["local_residual_delta"] >= 0.0
    assert diagnostics["recent_residual_delta"] >= 0.0
    assert "recurring_expert_delta" in diagnostics
    assert "transition_expert_delta" in diagnostics
    assert "high_risk_expert_delta" in diagnostics
    assert diagnostics["decision_threshold"] < 0.5


def test_regime_aware_delayed_bandit_rank_correction_can_create_per_example_spread():
    reference = TabularReferenceProfile(
        feature_mean=np.zeros(16, dtype=np.float32),
        feature_variance=np.ones(16, dtype=np.float32),
        mean_entropy=0.4,
        mean_probability=0.05,
        positive_rate=0.05,
        mean_confidence=0.8,
    )
    policy = RegimeAwareDelayedBanditTabularPolicy(reference, allowed_actions=("none",))
    features = np.zeros((4, 16), dtype=np.float32)
    features[:, 0] = np.array([2.0, 1.5, -1.0, -1.5], dtype=np.float32)
    labels = np.array([1, 1, 0, 0], dtype=np.int64)
    batch = TabularBatch(features=features, labels=labels, regime="rank_spread")
    signal = TabularShiftSignal(
        score=0.6,
        feature_score=0.2,
        output_score=0.1,
        collapse_risk=0.0,
        alert=False,
        severe=False,
        mean_entropy=0.35,
        mean_probability=0.10,
        positive_rate=0.05,
        mean_confidence=0.82,
    )
    risk_state = RiskState(raw_score=0.0, p_value=1.0, e_value=1.0, capital=1.0, alert=False)
    base_probabilities = [0.10, 0.10, 0.10, 0.10]

    class DummyModel:
        def predict_proba(self, _features):
            return list(base_probabilities)

    decision = policy.apply(DummyModel(), signal, risk_state, batch, base_probabilities)  # type: ignore[arg-type]
    feedback_state = policy.capture_feedback_state()
    assert feedback_state is not None
    before = np.asarray(policy.correct_probabilities(base_probabilities, signal=signal, risk_state=risk_state, batch=batch))
    policy.observe_delayed_outcome(
        feedback_state=feedback_state,
        model=DummyModel(),  # type: ignore[arg-type]
        batch=batch,
        signal=signal,
        risk_state=risk_state,
        decision=decision,
        batch_accuracy=0.5,
        reliability=0.8,
        utility=0.5,
        retrospective_reward=0.4,
        revealed_accuracy=0.5,
        revealed_coverage=1.0,
        revealed_baseline_accuracy=0.5,
        pending_delay_steps=8,
        pending_outstanding_count=3,
        revealed_mean_residual=0.0,
        predicted_positive_rate=0.10,
        revealed_positive_rate=0.50,
    )
    after = np.asarray(policy.correct_probabilities(base_probabilities, signal=signal, risk_state=risk_state, batch=batch))
    assert np.std(after - before) > 0.0
    assert after[0] > after[-1]
    diagnostics = policy.get_diagnostics()
    assert diagnostics["rank_update_rate"] > 0.0
    assert diagnostics["rank_delta_std"] > 0.0


def test_regime_aware_delayed_bandit_policy_state_round_trip_preserves_local_residuals():
    reference = TabularReferenceProfile(
        feature_mean=np.zeros(16, dtype=np.float32),
        feature_variance=np.ones(16, dtype=np.float32),
        mean_entropy=0.4,
        mean_probability=0.5,
        positive_rate=0.5,
        mean_confidence=0.8,
    )
    policy = RegimeAwareDelayedBanditTabularPolicy(reference, allowed_actions=("none",))
    batch = TabularBatch(
        features=np.random.randn(8, 16).astype(np.float32),
        labels=np.ones(8, dtype=np.int64),
        regime="delayed_residual_round_trip",
    )
    signal = TabularShiftSignal(
        score=0.4,
        feature_score=0.1,
        output_score=0.05,
        collapse_risk=0.0,
        alert=False,
        severe=False,
        mean_entropy=0.35,
        mean_probability=0.45,
        positive_rate=0.5,
        mean_confidence=0.82,
    )
    risk_state = RiskState(raw_score=0.0, p_value=1.0, e_value=1.0, capital=1.0, alert=False)

    class DummyModel:
        pass

    decision = policy.apply(DummyModel(), signal, risk_state, batch, [0.45] * len(batch.labels))  # type: ignore[arg-type]
    feedback_state = policy.capture_feedback_state()
    assert feedback_state is not None
    policy.observe_delayed_outcome(
        feedback_state=feedback_state,
        model=DummyModel(),  # type: ignore[arg-type]
        batch=batch,
        signal=signal,
        risk_state=risk_state,
        decision=decision,
        batch_accuracy=1.0,
        reliability=0.9,
        utility=0.9,
        retrospective_reward=0.9,
        revealed_accuracy=1.0,
        revealed_coverage=1.0,
        revealed_baseline_accuracy=0.5,
        pending_delay_steps=6,
        pending_outstanding_count=2,
        revealed_mean_residual=0.55,
        predicted_positive_rate=0.45,
        revealed_positive_rate=1.0,
    )
    state = export_policy_state(policy)
    clone = RegimeAwareDelayedBanditTabularPolicy(reference, allowed_actions=("none",))
    load_policy_state(clone, state)
    assert clone._residual_recent_bias == pytest.approx(policy._residual_recent_bias)
    assert clone._rank_bias == pytest.approx(policy._rank_bias)
    assert clone._rank_weights.shape == policy._rank_weights.shape
    assert clone._residual_prototype_bias.keys() == policy._residual_prototype_bias.keys()
    assert clone._residual_prototype_weights.keys() == policy._residual_prototype_weights.keys()
    assert clone._residual_expert_weights.keys() == policy._residual_expert_weights.keys()
    assert clone._residual_expert_bias.keys() == policy._residual_expert_bias.keys()
    assert clone._threshold_bias == pytest.approx(policy._threshold_bias)
    assert clone._threshold_prototype_bias.keys() == policy._threshold_prototype_bias.keys()
    assert clone._threshold_expert_bias.keys() == policy._threshold_expert_bias.keys()


def test_torch_tabular_supervised_head_update_changes_probabilities():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        random_state=0,
        stratify=data.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=7)
    model.fit_source(x_train, y_train, x_test[:64], y_test[:64], epochs=3)
    before = np.asarray(model.predict_proba(x_test[:32]))
    model.supervised_head_update(
        x_test[:32],
        y_test[:32],
        learning_rate=0.01,
        anchor_strength=0.65,
        max_parameter_drift=0.32,
        steps=1,
    )
    after = np.asarray(model.predict_proba(x_test[:32]))
    assert not np.allclose(before, after)


def test_torch_tabular_supervised_head_adapter_update_changes_probabilities_with_bounded_drift():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        random_state=0,
        stratify=data.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=11)
    model.fit_source(x_train, y_train, x_test[:64], y_test[:64], epochs=3)
    before = np.asarray(model.predict_proba(x_test[:32]))
    model.supervised_head_adapter_update(
        x_test[:32],
        y_test[:32],
        learning_rate=0.008,
        anchor_strength=0.78,
        max_parameter_drift=0.32,
        steps=2,
    )
    after = np.asarray(model.predict_proba(x_test[:32]))
    assert not np.allclose(before, after)
    assert model.parameter_drift() <= 0.320001


def test_torch_tabular_supervised_trusted_subspace_update_changes_probabilities_with_bounded_drift():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        random_state=0,
        stratify=data.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=13)
    model.fit_source(x_train, y_train, x_test[:64], y_test[:64], epochs=3)
    before = np.asarray(model.predict_proba(x_test[:32]))
    update_fraction = model.supervised_trusted_subspace_update(
        x_test[:32],
        y_test[:32],
        learning_rate=0.007,
        anchor_strength=0.82,
        max_parameter_drift=0.32,
        confidence_threshold=0.72,
        min_selected=6,
        subspace_fraction=0.35,
        steps=2,
    )
    after = np.asarray(model.predict_proba(x_test[:32]))
    assert update_fraction > 0.0
    assert not np.allclose(before, after)
    assert model.parameter_drift() <= 0.320001


def test_torch_tabular_pairwise_head_adapter_update_improves_positive_negative_separation():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data,
        data.target,
        random_state=0,
        stratify=data.target,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)
    model = TorchTabularAdapterModel(input_dim=x_train.shape[1], seed=17)
    model.fit_source(x_train, y_train, x_test[:64], y_test[:64], epochs=3)
    eval_features = x_test[:48]
    eval_labels = y_test[:48]
    before = np.asarray(model.predict_proba(eval_features))
    before_gap = float(before[eval_labels == 1].mean() - before[eval_labels == 0].mean())
    update_fraction = model.supervised_pairwise_head_adapter_update(
        eval_features,
        eval_labels,
        learning_rate=0.006,
        anchor_strength=0.72,
        max_parameter_drift=0.34,
        segment_ids=np.digitize(before, bins=np.array([0.25, 0.65], dtype=np.float64)),
        classification_weight=0.25,
        margin=0.05,
        max_pairs=96,
        steps=2,
    )
    after = np.asarray(model.predict_proba(eval_features))
    after_gap = float(after[eval_labels == 1].mean() - after[eval_labels == 0].mean())
    assert update_fraction > 0.0
    assert after_gap >= before_gap
    assert model.parameter_drift() <= 0.340001


def test_fraud_rank_delayed_bandit_policy_persists_segment_and_pairwise_state():
    reference = TabularReferenceProfile(
        feature_mean=np.zeros(16, dtype=np.float32),
        feature_variance=np.ones(16, dtype=np.float32),
        mean_entropy=0.4,
        mean_probability=0.05,
        positive_rate=0.05,
        mean_confidence=0.92,
    )
    policy = FraudRankDelayedBanditTabularPolicy(reference, allowed_actions=("none",))
    batch = TabularBatch(
        features=np.random.randn(6, 16).astype(np.float32),
        labels=np.array([1, 0, 1, 0, 0, 0], dtype=np.int64),
        regime="fraud_rank",
    )
    policy._rank_weights[:4] = np.array([0.1, -0.2, 0.05, 0.03])
    policy._rank_bias = 0.11
    policy._pairwise_rank_update_rate = 0.4
    state = export_policy_state(policy)
    clone = FraudRankDelayedBanditTabularPolicy(reference, allowed_actions=("none",))
    load_policy_state(clone, state)
    assert clone._fraud_rank_mode is True
    assert clone._segment_count == policy._segment_count
    assert clone._rank_weights.shape == policy._rank_weights.shape
    assert clone._rank_bias == pytest.approx(policy._rank_bias)
    segment_ids = clone._rank_segment_ids(
        features=batch.features,
        probabilities=np.asarray([0.04, 0.11, 0.62, 0.19, 0.08, 0.41], dtype=np.float64),
    )
    assert len(np.unique(segment_ids)) >= 2


def test_audit_store_records_interventions(trained_layer: ReliabilityLayer):
    batch = RuntimeBatch(features=np.random.randn(16, 30).astype(np.float32), labels=np.zeros(16, dtype=np.int64))
    trained_layer.process_batch(batch)
    records = trained_layer.governance.audit.fetch_recent(limit=5)
    assert len(records) == 1
    assert records[0]["recommended_action"]


def test_snapshot_store_and_rollback(trained_layer: ReliabilityLayer, temp_governance_dirs):
    adapter = trained_layer._adapter
    before = adapter.export_snapshot()
    snapshot_id = trained_layer.governance.snapshots.save(adapter, reason="test", step=0)
    adapter.reset()
    trained_layer.rollback(snapshot_id, actor="tester")
    after = adapter.export_snapshot()
    assert before.payload.temperature == after.payload.temperature


def test_deployment_surface_to_dict(trained_layer: ReliabilityLayer):
    batch = RuntimeBatch(features=np.random.randn(16, 30).astype(np.float32))
    surface = trained_layer.process_batch(batch)
    payload = surface.to_dict()
    assert "shift_score" in payload
    assert "recommended_action" in payload
    assert payload["operating_mode"] == "shadow"
    decision_record = surface.decision_record()
    assert "regime_id" in decision_record
    assert "retrain_recommended" in decision_record
    assert "shift_signature" in decision_record
    assert "controller_profile" in decision_record
    assert "adaptation_opportunity_score" in decision_record
    assert "monitor_saturated" in decision_record


def test_sklearn_adapter_predict_proba():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    estimator = LogisticRegression(max_iter=200).fit(x_train, y_train)
    adapter = SklearnModelAdapter(
        estimator,
        source_feature_mean=x_train.mean(axis=0),
        source_feature_std=np.clip(x_train.std(axis=0), 1e-3, None),
        source_positive_rate=float(y_train.mean()),
    )
    probabilities = adapter.predict_proba(x_test[:8])
    assert len(probabilities) == 8
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_sklearn_adapter_refresh_and_adapt_change_probabilities():
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, random_state=0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    estimator = LogisticRegression(max_iter=200).fit(x_train, y_train)
    adapter = SklearnModelAdapter(
        estimator,
        source_feature_mean=x_train.mean(axis=0),
        source_feature_std=np.clip(x_train.std(axis=0), 1e-3, None),
        source_positive_rate=float(y_train.mean()),
    )
    shifted = (x_test[:32] * 1.35 + 0.25).astype(np.float32)
    before = np.asarray(adapter.predict_proba(shifted))
    adapter.refresh_batch_norm(shifted, passes=2)
    refreshed = np.asarray(adapter.predict_proba(shifted))
    selected_fraction = adapter.adapt(
        shifted,
        refreshed.tolist(),
        learning_rate=0.03,
        confidence_threshold=0.75,
        anchor_strength=0.15,
        entropy_weight=0.08,
        max_parameter_drift=0.8,
        steps=2,
    )
    after = np.asarray(adapter.predict_proba(shifted))
    assert not np.allclose(before, refreshed)
    assert not np.allclose(refreshed, after)
    assert selected_fraction >= 0.0


def test_black_box_adapter():
    def predict_proba(features: np.ndarray) -> np.ndarray:
        return np.full(len(features), 0.7)

    adapter = BlackBoxModelAdapter(predict_proba)
    assert adapter.predict_proba(np.zeros((3, 4))) == [0.7, 0.7, 0.7]
    assert adapter.supports_adaptation is False


def test_apply_operating_mode_blocks_high_risk():
    class DummyAdapter:
        def load_snapshot(self, snapshot):
            self.loaded = snapshot

        loaded = None

    adapter = DummyAdapter()
    decision = InterventionDecision(action="reset", reason="test")
    action_taken, reason = apply_operating_mode(
        mode=OperatingMode.BOUNDED_AUTO,
        bounded_auto_actions=frozenset({"none", "hold"}),
        adapter=adapter,  # type: ignore[arg-type]
        decision=decision,
        snapshot_before="snap",
    )
    assert action_taken == "none"
    assert "blocked" in reason


def test_replay_stream_batches():
    stream = build_synthetic_fraud_like_stream(steps=4, batch_size=8)
    batches = list(iter_replay_batches(stream, batch_size=8, max_steps=4))
    assert len(batches) == 4


def test_offline_replay_comparison(runtime_config: RuntimeConfig):
    stream = build_synthetic_fraud_like_stream(steps=4, batch_size=16)
    result = run_offline_replay_comparison(
        stream,
        runtime_config=runtime_config,
        strategies=("frozen", "controller"),
    )
    assert len(result.summaries) == 2
    assert result.summaries[0].steps == 4


def test_pilot_case_study(runtime_config: RuntimeConfig, tmp_path: Path):
    summary = run_pilot_case_study(DEFAULT_PILOT, runtime_config=runtime_config, output_dir=tmp_path / "pilot")
    assert Path(summary["report_md"]).exists()
    assert Path(summary["report_json"]).exists()


def test_recommend_approval_uses_pending_decision(runtime_config: RuntimeConfig):
    config = replace(runtime_config, operating_mode=OperatingMode.RECOMMEND)
    layer = build_layer_for_tabular_replay(config=config)

    class FixedPolicy:
        def __init__(self) -> None:
            self.calls = 0

        def apply(self, model, signal, risk_state, batch, probabilities):
            del model, signal, risk_state, batch, probabilities
            self.calls += 1
            return TabularDecision(action="label_shift", reason="force_label_shift")

    policy = FixedPolicy()
    layer._policy = policy  # type: ignore[attr-defined]
    batch = RuntimeBatch(
        features=np.random.randn(16, 30).astype(np.float32),
        labels=np.zeros(16, dtype=np.int64),
        regime="approval_test",
    )

    recommended = layer.process_batch(batch)
    assert recommended.recommended_action == "label_shift"
    assert recommended.action_taken == "none"

    approved = layer.approve_and_apply(batch, approved_action="label_shift", approver="tester")
    assert policy.calls == 1
    assert approved.action_taken == "label_shift"
    assert approved.step == recommended.step
    assert layer._step == recommended.step + 1  # type: ignore[attr-defined]


def test_real_data_bundle_build_layer_returns_fresh_adapter(runtime_config: RuntimeConfig):
    from adaptive_reliability_layer.replay.real_data import load_breast_cancer_bundle

    bundle = load_breast_cancer_bundle(steps=4, batch_size=16, seed=3)
    layer1 = bundle.build_layer(runtime_config)
    layer2 = bundle.build_layer(runtime_config)
    assert layer1._adapter is not layer2._adapter  # type: ignore[attr-defined]

    drift_before = layer2._adapter.parameter_drift()  # type: ignore[attr-defined]
    layer1._adapter.recalibrate_temperature(  # type: ignore[attr-defined]
        reference_confidence=0.80,
        observed_confidence=0.55,
    )
    assert layer1._adapter.parameter_drift() > 0.0  # type: ignore[attr-defined]
    assert layer2._adapter.parameter_drift() == drift_before  # type: ignore[attr-defined]


def test_delayed_label_replay_scores_original_batch_labels():
    from adaptive_reliability_layer.replay.loader import ReplayRecord, ReplayStream
    from adaptive_reliability_layer.runtime.config import ReplayConfig

    class DummyLayer:
        def __init__(self) -> None:
            self.step = 0

        def process_batch(self, batch: RuntimeBatch) -> DeploymentSurface:
            predictions = [1 if row[0] > 0 else 0 for row in batch.features]
            probabilities = [0.9 if prediction == 1 else 0.1 for prediction in predictions]
            surface = DeploymentSurface(
                step=self.step,
                predictions=predictions,
                probabilities=probabilities,
                confidence=0.9,
                shift_score=0.0,
                feature_shift_score=0.0,
                output_shift_score=0.0,
                collapse_risk=0.0,
                risk_capital=1.0,
                risk_alert=False,
                regime_hint=batch.regime,
                recommended_action="none",
                action_taken="none",
                intervention_reason="dummy",
                trust_state="normal",
                reliability_score=1.0,
                parameter_drift=0.0,
                operating_mode="shadow",
                model_version="dummy-v1",
                specialist_id=None,
                rollback_available=False,
                snapshot_id=None,
                abstained=False,
            )
            self.step += 1
            return surface

    stream = ReplayStream(
        records=(
            ReplayRecord(timestamp="t0", features=np.array([-1.0], dtype=np.float32), label=0, metadata={}),
            ReplayRecord(timestamp="t1", features=np.array([-2.0], dtype=np.float32), label=0, metadata={}),
            ReplayRecord(timestamp="t2", features=np.array([1.0], dtype=np.float32), label=1, metadata={}),
            ReplayRecord(timestamp="t3", features=np.array([2.0], dtype=np.float32), label=1, metadata={}),
        ),
        feature_columns=("feature_0",),
    )
    state = run_replay_on_stream(
        DummyLayer(),  # type: ignore[arg-type]
        stream,
        config=ReplayConfig(batch_size=2, label_delay_steps=1, max_steps=2),
        name="dummy",
    )
    assert state.accuracies == [1.0, 1.0]
    assert state.utilities == [1.0, 1.0]


def test_pilot_cli_allows_zero_label_delay(monkeypatch, tmp_path: Path):
    from adaptive_reliability_layer import cli

    captured: dict[str, int] = {}

    def fake_run_pilot_case_study(pilot, *, runtime_config, output_dir, layer_builder=None):
        del runtime_config, output_dir, layer_builder
        captured["delay"] = pilot.label_delay_steps
        return {
            "report_md": str(tmp_path / "pilot.md"),
            "report_json": str(tmp_path / "pilot.json"),
            "dataset_csv": str(tmp_path / "stream.csv"),
            "primary_kpi": "utility_under_delayed_labels",
            "utility_delta": 0.0,
            "risk_reduction": 0.0,
        }

    monkeypatch.setattr(cli, "run_pilot_case_study", fake_run_pilot_case_study)
    config_path = Path(__file__).resolve().parents[1] / "configs" / "pilot_fraud_tabular.yaml"
    cli.pilot_main(
        [
            "--config",
            str(config_path),
            "--label-delay-steps",
            "0",
            "--output-dir",
            str(tmp_path / "pilot"),
        ]
    )
    assert captured["delay"] == 0


def test_csv_roundtrip(tmp_path: Path, runtime_config: RuntimeConfig):
    from adaptive_reliability_layer.replay.engine import export_stream_to_csv

    stream = build_synthetic_fraud_like_stream(steps=2, batch_size=8)
    csv_path = tmp_path / "stream.csv"
    export_stream_to_csv(stream, csv_path)
    loaded = load_replay_csv(csv_path, runtime_config.replay)
    assert len(loaded.records) == len(stream.records)
    assert loaded.records[0].metadata.get("regime") is not None


def test_real_data_breast_cancer_bundle(runtime_config: RuntimeConfig):
    from adaptive_reliability_layer.replay.real_data import load_breast_cancer_bundle
    from adaptive_reliability_layer.replay.verification_suite import verify_real_data_source

    bundle = load_breast_cancer_bundle(steps=4, batch_size=16)
    result = verify_real_data_source(bundle, runtime_config=runtime_config, strategies=("frozen", "controller"))
    assert result.source_id == "sklearn_breast_cancer"
    assert len(result.priority_checks) == 8
    assert result.replay.controller_vs_frozen_risk_reduction is not None


@pytest.mark.slow
def test_real_data_verification_fast_sources(runtime_config: RuntimeConfig, tmp_path: Path):
    from adaptive_reliability_layer.replay.verification_suite import run_real_data_verification_suite

    suite = run_real_data_verification_suite(
        runtime_config=runtime_config,
        source_ids=("breast_cancer", "digits", "tabular_breast_cancer_shift"),
        output_dir=tmp_path / "verify",
    )
    assert len(suite.sources) == 3
    assert (tmp_path / "verify" / "verification_suite.md").exists()


def test_openml_electricity_bundle_is_time_ordered():
    from adaptive_reliability_layer.replay.real_data import load_openml_electricity_bundle

    bundle = load_openml_electricity_bundle(steps=6, batch_size=16, stream_cycles=2)
    assert bundle.stream.records
    assert bundle.stream.records[0].metadata.get("time_ordered") is True
    assert bundle.stream.records[0].metadata.get("controller_profile") == "sensor_safe"
