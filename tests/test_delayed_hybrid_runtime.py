from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay, run_replay_on_stream
from adaptive_reliability_layer.replay.engine import build_synthetic_fraud_like_stream
from adaptive_reliability_layer.runtime.action_gating import build_runtime_policy
from adaptive_reliability_layer.runtime.config import RuntimeConfig, load_runtime_config
from adaptive_reliability_layer.runtime.policy_state import export_policy_state, load_policy_state
from adaptive_reliability_layer.runtime.types import OperatingMode, RuntimeBatch
from adaptive_reliability_layer.tabular_benchmark import (
    DelayedHybridBanditSpecialistPolicy,
    FraudContextDelayedBanditTabularPolicy,
    FraudRankDelayedBanditTabularPolicy,
    TabularBatch,
    TabularReferenceProfile,
    TabularShiftSignal,
)
from adaptive_reliability_layer.risk import RiskState


@pytest.fixture
def reference_profile() -> TabularReferenceProfile:
    return TabularReferenceProfile(
        feature_mean=np.zeros(16, dtype=np.float32),
        feature_variance=np.ones(16, dtype=np.float32),
        mean_entropy=0.4,
        mean_probability=0.5,
        positive_rate=0.5,
        mean_confidence=0.8,
    )


def test_build_runtime_policy_delayed_hybrid(reference_profile: TabularReferenceProfile):
    policy = build_runtime_policy(
        "delayed_hybrid",
        reference_profile,
        type("Cfg", (), {"bandit_alpha": 0.8, "allowed_actions": ("none", "reset"), "max_specialists": 3, "distance_threshold": 1.1})(),
    )
    assert isinstance(policy, DelayedHybridBanditSpecialistPolicy)


def test_build_runtime_policy_fraud_rank(reference_profile: TabularReferenceProfile):
    policy = build_runtime_policy(
        "fraud_rank_delayed_bandit",
        reference_profile,
        type("Cfg", (), {"bandit_alpha": 0.8, "allowed_actions": ("none", "reset")})(),
    )
    assert isinstance(policy, FraudRankDelayedBanditTabularPolicy)


def test_build_runtime_policy_fraud_context(reference_profile: TabularReferenceProfile):
    policy = build_runtime_policy(
        "fraud_context_delayed_bandit",
        reference_profile,
        type("Cfg", (), {"bandit_alpha": 0.8, "allowed_actions": ("none", "reset")})(),
    )
    assert isinstance(policy, FraudContextDelayedBanditTabularPolicy)


def test_delayed_hybrid_forwards_pending_summary_and_residual_correction(reference_profile: TabularReferenceProfile):
    policy = DelayedHybridBanditSpecialistPolicy(
        reference_profile,
        controller_kwargs={"allowed_actions": ("none",)},
    )
    batch = TabularBatch(
        features=np.random.randn(8, 16).astype(np.float32),
        labels=np.ones(8, dtype=np.int64),
        regime="hybrid",
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
        def export_state(self):
            return {}

        def load_state(self, _state):
            return None

        def predict_proba(self, _features):
            return [0.45] * len(batch.labels)

    model = DummyModel()
    policy.prepare_model(model, batch)  # type: ignore[arg-type]
    decision = policy.apply(model, signal, risk_state, batch, [0.45] * len(batch.labels))  # type: ignore[arg-type]
    feedback_state = policy.capture_feedback_state(
        model=model,  # type: ignore[arg-type]
        batch=batch,
        signal=signal,
        risk_state=risk_state,
        decision=decision,
    )
    assert feedback_state is not None

    policy.update_pending_feedback_summary(
        pending_count=3,
        mean_age=5.0,
        max_age=8.0,
        stale_fraction=0.33,
    )
    before = policy.correct_probabilities([0.45, 0.45], signal=signal, risk_state=risk_state, batch=batch)
    controller = policy._active_controller()
    controller.observe_delayed_outcome(  # type: ignore[attr-defined]
        feedback_state=feedback_state.inner_feedback_state,
        model=model,  # type: ignore[arg-type]
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


def test_reliability_layer_replay_with_delayed_hybrid(tmp_path):
    base = load_runtime_config("configs/default.yaml")
    config = RuntimeConfig(
        operating_mode=OperatingMode.BOUNDED_AUTO,
        bounded_auto_actions=base.bounded_auto_actions,
        model_version="test-delayed-hybrid",
        monitor=base.monitor,
        policy=replace(
            base.policy,
            name="delayed_hybrid",
            allowed_actions=("none", "hold", "recalibrate", "label_shift", "reset"),
            distance_threshold=0.55,
        ),
        governance=base.governance.__class__(
            audit_db_path=str(tmp_path / "audit.db"),
            snapshot_dir=str(tmp_path / "snapshots"),
            max_snapshots=5,
            policy_version="test",
            environment="test",
        ),
        metrics=base.metrics.__class__(enabled=False, prometheus_port=9091, namespace="arl_test"),
        replay=replace(base.replay, label_delay_steps=2, max_steps=6, batch_size=32),
        sota=base.sota,
        log_json=False,
    )
    layer = build_layer_for_tabular_replay(config=config)
    stream = build_synthetic_fraud_like_stream(steps=6, batch_size=32)
    state = run_replay_on_stream(layer, stream, config=config.replay, name="delayed_hybrid")
    assert len(state.surfaces) == 6
    assert layer.pending_delayed_count >= 0
    diagnostics = layer._policy.get_diagnostics()
    assert "specialist_route_reuses" in diagnostics or "specialist_last_exchangeability_score" in diagnostics


def test_delayed_hybrid_policy_state_round_trip(reference_profile: TabularReferenceProfile):
    policy = DelayedHybridBanditSpecialistPolicy(reference_profile, controller_kwargs={"allowed_actions": ("none",)})
    batch = TabularBatch(
        features=np.random.randn(4, 16).astype(np.float32),
        labels=np.array([0, 1, 0, 1], dtype=np.int64),
        regime="state_round_trip",
    )

    class DummyModel:
        def export_state(self):
            return {"weights": [1.0]}

        def load_state(self, _state):
            return None

    model = DummyModel()
    policy.prepare_model(model, batch)  # type: ignore[arg-type]
    exported = export_policy_state(policy)
    assert exported["kind"] == "delayed_hybrid"
    # New format: specialists (includes snapshot + metadata); old key was "controllers"
    assert "specialists" in exported
    assert len(exported["specialists"]) >= 1
    # Each specialist entry must have the key fields
    for slot in exported["specialists"]:
        assert "name" in slot
        assert "signature" in slot
        assert "controller" in slot
        assert "creation_positive_rate" in slot

    clone = DelayedHybridBanditSpecialistPolicy(reference_profile, controller_kwargs={"allowed_actions": ("none",)})
    clone.prepare_model(model, batch)  # type: ignore[arg-type]
    load_policy_state(clone, exported)
    assert clone._active_index == policy._active_index
    assert len(clone._specialists) == len(policy._specialists)
