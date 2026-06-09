from __future__ import annotations

import numpy as np
import pytest

from adaptive_reliability_layer.replay.production_benchmark import (
    ProductionBenchmarkSpec,
    ProductionEvidenceThresholds,
    ProductionSourceSpec,
    evaluate_source_evidence,
    load_production_benchmark_spec,
    _merged_strategies,
)
from adaptive_reliability_layer.replay.engine import build_synthetic_fraud_like_stream
from adaptive_reliability_layer.replay.real_data import (
    RealDataBundle,
    _split_train_test_indices,
    load_paysim_fraud_bundle,
)
from adaptive_reliability_layer.replay.dual_metric import run_dual_mode_replay
from adaptive_reliability_layer.replay.report import (
    ReplayComparisonResult,
    ReplayTimeline,
    StrategyReplaySummary,
    utility_delta_vs_baseline,
)
from adaptive_reliability_layer.runtime.config import load_runtime_config
from adaptive_reliability_layer.runtime.sota.regime_coreset import RegimeCoreset, ReservoirClusterRouter
from adaptive_reliability_layer.runtime.types import DeploymentSurface, OperatingMode
from adaptive_reliability_layer.tabular_benchmark import ScheduledRetrainTabularPolicy


def test_load_production_benchmark_spec():
    runtime, spec = load_production_benchmark_spec("configs/production_benchmark_suite.yaml")
    assert runtime.operating_mode == OperatingMode.BOUNDED_AUTO
    assert spec.controller_name == "regime_aware_delayed_bandit"
    assert "scheduled_retrain" in spec.baseline_strategies
    assert any(source.id == "ulb_creditcard_fraud" for source in spec.sources)
    assert any(source.temporal_split for source in spec.sources)


def test_load_sota_production_benchmark_spec():
    runtime, spec = load_production_benchmark_spec("configs/production_benchmark_sota_suite.yaml")
    assert spec.controller_name == "delayed_hybrid"
    assert spec.require_beat_baselines is True
    assert runtime.sota.enabled is True
    assert runtime.sota.asr_reset_enabled is True


def test_merged_strategies_deduplicates():
    spec = ProductionBenchmarkSpec(
        controller_name="delayed_hybrid",
        strategies=("frozen", "delayed_hybrid"),
        baseline_strategies=("scheduled_retrain", "naive", "frozen"),
    )
    merged = _merged_strategies(spec)
    assert merged[0] == "frozen"
    assert merged.count("frozen") == 1
    assert "delayed_hybrid" in merged
    assert "scheduled_retrain" in merged


def test_regime_coreset_keeps_diverse_points():
    coreset = RegimeCoreset(max_size=4)
    features = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]], dtype=np.float32)
    labels = np.array([0, 1, 0, 1, 1], dtype=np.int64)
    coreset.update(batch_features=features, batch_labels=labels, batch_utility=0.8)
    assert coreset.support_features() is not None
    assert coreset.support_features().shape[0] <= 4


def test_reservoir_cluster_router_assigns():
    router = ReservoirClusterRouter(max_clusters=3)
    first = router.assign(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    second = router.assign(np.array([0.11, 0.19, 0.31], dtype=np.float32))
    far = router.assign(np.array([5.0, 5.0, 5.0], dtype=np.float32))
    assert first == second
    assert far != first


def test_temporal_split_orders_train_before_test():
    labels = np.arange(100)
    time_rank = labels.copy()
    train_idx, test_idx = _split_train_test_indices(
        labels,
        time_rank,
        test_fraction=0.25,
        seed=7,
        temporal_split=True,
    )
    assert train_idx.max() < test_idx.min()


def test_scheduled_retrain_policy_triggers_adapt():
    policy = ScheduledRetrainTabularPolicy(retrain_interval=2)
    from adaptive_reliability_layer.torch_model import TorchTabularAdapterModel

    model = TorchTabularAdapterModel(4, seed=3)
    x = np.random.default_rng(0).normal(size=(8, 4)).astype(np.float32)
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    model.fit_source(x, y, x[:2], y[:2], epochs=1)
    batch = type("Batch", (), {"features": x[:4], "labels": y[:4], "regime": "test"})()
    from adaptive_reliability_layer.tabular_benchmark import TabularShiftSignal

    signal = TabularShiftSignal(
        score=1.0,
        feature_score=0.2,
        output_score=0.2,
        collapse_risk=0.1,
        alert=False,
        severe=False,
        mean_entropy=0.5,
        mean_probability=0.5,
        positive_rate=0.5,
        mean_confidence=0.7,
    )
    risk = type("Risk", (), {"capital": 1.0, "alert": False})()
    probs = model.predict_proba(batch.features)
    first = policy.apply(model, signal, risk, batch, probs)
    assert first.action == "none"
    second = policy.apply(model, signal, risk, batch, probs)
    assert second.action in {"adapt", "recalibrate"}


def test_adaptation_safety_allows_light_mutations_under_structural_shift():
    from adaptive_reliability_layer.runtime.sota.adaptation_safety import AdaptationSafetyTracker

    tracker = AdaptationSafetyTracker()
    assert tracker.record(
        step=1,
        operating_mode="bounded_auto",
        action_taken="recalibrate",
        collapse_risk=0.70,
        parameter_drift=0.12,
        force_shadow=False,
        shift_score=2.0,
    )
    assert tracker.record(
        step=2,
        operating_mode="bounded_auto",
        action_taken="recalibrate",
        collapse_risk=0.70,
        parameter_drift=0.12,
        force_shadow=False,
        shift_score=1.0,
    ) is False


def test_policy_activity_correction_pass():
    thresholds = ProductionEvidenceThresholds(min_stream_records=100, min_utility_delta=0.005)
    source = ProductionSourceSpec(id="demo", tier="core")
    replay = ReplayComparisonResult(
        summaries=(
            StrategyReplaySummary(
                name="frozen",
                steps=10,
                mean_accuracy=0.9,
                mean_utility=0.9,
                mean_risk_capital=10.0,
                mean_shift_score=1.0,
                mean_reliability=0.8,
                intervention_rate=0.0,
                reset_count=0,
                risk_alert_count=5,
                first_drift_step=None,
                recommendation_rate=0.0,
                recommendation_execution_rate=0.0,
                bounded_interventions_per_1000=0.0,
                retrain_recommendation_count=0,
                first_retrain_recommendation_step=None,
                budget_limited_count=0,
            ),
            StrategyReplaySummary(
                name="delayed_hybrid",
                steps=10,
                mean_accuracy=0.92,
                mean_utility=0.92,
                mean_risk_capital=8.0,
                mean_shift_score=1.0,
                mean_reliability=0.85,
                intervention_rate=0.0,
                reset_count=0,
                risk_alert_count=3,
                first_drift_step=None,
                recommendation_rate=0.0,
                recommendation_execution_rate=0.0,
                bounded_interventions_per_1000=0.0,
                retrain_recommendation_count=0,
                first_retrain_recommendation_step=None,
                budget_limited_count=0,
                correction_applied_rate=0.9,
                correction_only_rate=0.9,
                mean_correction_flipped_predictions=0.2,
            ),
            StrategyReplaySummary(
                name="scheduled_retrain",
                steps=10,
                mean_accuracy=0.91,
                mean_utility=0.91,
                mean_risk_capital=9.0,
                mean_shift_score=1.0,
                mean_reliability=0.82,
                intervention_rate=0.5,
                reset_count=0,
                risk_alert_count=4,
                first_drift_step=None,
                recommendation_rate=0.5,
                recommendation_execution_rate=0.0,
                bounded_interventions_per_1000=50.0,
                retrain_recommendation_count=0,
                first_retrain_recommendation_step=None,
                budget_limited_count=0,
            ),
        ),
        controller_vs_frozen_utility_delta=0.02,
        controller_vs_frozen_risk_reduction=0.0,
        controller_vs_frozen_harmful_events_avoided=0,
        controller_vs_frozen_retrain_deferral_steps=0,
    )
    bundle = RealDataBundle(
        source_id="demo",
        wedge="fraud_risk",
        description="demo",
        adapter_kind="torch",
        feature_dim=8,
        train_size=100,
        stream_size=1000,
        stream=build_synthetic_fraud_like_stream(steps=4, batch_size=32),
        build_layer=lambda config: None,  # type: ignore[assignment]
        reference_batches=[],
        validation_accuracy=0.9,
    )
    result = evaluate_source_evidence(
        source=source,
        bundle=bundle,
        dual_payload={"modes": {"bounded_auto": {"replay": replay}}, "label_delay_steps": 4},
        thresholds=thresholds,
        controller_name="delayed_hybrid",
        baseline_strategies=("scheduled_retrain",),
        require_beat_baselines=True,
    )
    policy_check = next(check for check in result.checks if check.name == "policy_activity")
    assert policy_check.passed
    assert "correction_rate=0.900" in policy_check.detail


def test_evaluate_source_evidence_passes_on_positive_delta():
    thresholds = ProductionEvidenceThresholds(min_stream_records=100, min_utility_delta=0.01)
    source = ProductionSourceSpec(id="demo", tier="core")
    replay = ReplayComparisonResult(
        summaries=(
            StrategyReplaySummary(
                name="frozen",
                steps=10,
                mean_accuracy=0.9,
                mean_utility=0.9,
                mean_risk_capital=10.0,
                mean_shift_score=1.0,
                mean_reliability=0.8,
                intervention_rate=0.0,
                reset_count=0,
                risk_alert_count=5,
                first_drift_step=None,
                recommendation_rate=0.0,
                recommendation_execution_rate=0.0,
                bounded_interventions_per_1000=0.0,
                retrain_recommendation_count=0,
                first_retrain_recommendation_step=None,
                budget_limited_count=0,
            ),
            StrategyReplaySummary(
                name="regime_aware_delayed_bandit",
                steps=10,
                mean_accuracy=0.92,
                mean_utility=0.92,
                mean_risk_capital=8.0,
                mean_shift_score=1.0,
                mean_reliability=0.85,
                intervention_rate=0.2,
                reset_count=0,
                risk_alert_count=3,
                first_drift_step=None,
                recommendation_rate=0.5,
                recommendation_execution_rate=0.2,
                bounded_interventions_per_1000=20.0,
                retrain_recommendation_count=0,
                first_retrain_recommendation_step=None,
                budget_limited_count=0,
            ),
            StrategyReplaySummary(
                name="scheduled_retrain",
                steps=10,
                mean_accuracy=0.91,
                mean_utility=0.91,
                mean_risk_capital=9.0,
                mean_shift_score=1.0,
                mean_reliability=0.82,
                intervention_rate=0.3,
                reset_count=0,
                risk_alert_count=4,
                first_drift_step=None,
                recommendation_rate=0.0,
                recommendation_execution_rate=0.0,
                bounded_interventions_per_1000=30.0,
                retrain_recommendation_count=0,
                first_retrain_recommendation_step=None,
                budget_limited_count=0,
            ),
        ),
        controller_vs_frozen_utility_delta=0.02,
        controller_vs_frozen_risk_reduction=0.2,
        controller_vs_frozen_harmful_events_avoided=1,
        controller_vs_frozen_retrain_deferral_steps=2,
        timelines=(
            ReplayTimeline(name="frozen", surfaces=()),
            ReplayTimeline(
                name="regime_aware_delayed_bandit",
                surfaces=(
                    DeploymentSurface(
                        step=1,
                        predictions=[1],
                        probabilities=[0.9],
                        confidence=0.9,
                        shift_score=1.0,
                        feature_shift_score=0.2,
                        output_shift_score=0.2,
                        collapse_risk=0.1,
                        risk_capital=1.0,
                        risk_alert=False,
                        regime_hint="r",
                        recommended_action="adapt",
                        action_taken="adapt",
                        intervention_reason="test",
                        why_this_action="test",
                        trust_state="normal",
                        reliability_score=0.8,
                        parameter_drift=0.01,
                        operating_mode="bounded_auto",
                        effective_operating_mode="bounded_auto",
                        model_version="v1",
                        specialist_id=None,
                        rollback_available=True,
                        rollback_eligible=True,
                        snapshot_id="s1",
                        abstained=False,
                        regime_id="r",
                        regime_confidence=0.5,
                        regime_novelty=0.1,
                        risk_score=0.5,
                        recommended_action_requires_approval=False,
                        retrain_recommended=False,
                        budget_limited=False,
                        budget_reason=None,
                        batch_id="b1",
                        shift_signature="sig",
                        controller_profile="general",
                        adaptation_opportunity_score=0.5,
                        adaptation_safety_ok=True,
                    ),
                ),
            ),
        ),
    )
    assert utility_delta_vs_baseline(
        replay,
        controller_name="regime_aware_delayed_bandit",
        baseline_name="scheduled_retrain",
    ) == pytest.approx(0.01)

    bundle = RealDataBundle(
        source_id="demo",
        wedge="fraud_risk",
        description="demo",
        adapter_kind="sklearn",
        feature_dim=8,
        train_size=100,
        stream_size=1000,
        stream=build_synthetic_fraud_like_stream(steps=4, batch_size=32),
        build_layer=lambda config: None,  # type: ignore[assignment]
        reference_batches=[],
        validation_accuracy=0.9,
    )
    dual = {
        "modes": {"bounded_auto": {"replay": replay}},
        "label_delay_steps": 4,
        "temporal_split": True,
    }
    result = evaluate_source_evidence(
        source=source,
        bundle=bundle,
        dual_payload=dual,
        thresholds=thresholds,
        controller_name="regime_aware_delayed_bandit",
        baseline_strategies=("scheduled_retrain",),
        require_beat_baselines=False,
    )
    assert result.passed
    assert result.baseline_utility_deltas[0][1] == pytest.approx(0.01)


@pytest.mark.slow
def test_production_source_smoke_paysim(tmp_path):
    runtime = load_runtime_config("configs/production_benchmark_suite.yaml")
    bundle = load_paysim_fraud_bundle(steps=4, batch_size=32, stream_cycles=1, temporal_split=True)
    spec = ProductionBenchmarkSpec(
        controller_name="regime_aware_delayed_bandit",
        strategies=("frozen", "regime_aware_delayed_bandit"),
        baseline_strategies=("scheduled_retrain", "naive"),
        evidence=ProductionEvidenceThresholds(min_stream_records=100),
        sources=(ProductionSourceSpec(id="paysim_fraud", tier="auxiliary", steps=4, batch_size=32),),
    )
    from dataclasses import replace
    from adaptive_reliability_layer.replay.production_benchmark import run_production_source_benchmark

    replay_config = replace(runtime.replay, label_delay_steps=2, max_steps=4, batch_size=32)
    source_config = replace(
        runtime,
        replay=replay_config,
        governance=replace(
            runtime.governance,
            audit_db_path=str(tmp_path / "audit.db"),
            snapshot_dir=str(tmp_path / "snapshots"),
        ),
    )
    _, _, result = run_production_source_benchmark(
        source=spec.sources[0],
        runtime_config=source_config,
        spec=spec,
        bundle_loader=lambda **kwargs: load_paysim_fraud_bundle(**kwargs),
    )
    assert result.stream_records > 0
    assert result.utility_delta is not None
