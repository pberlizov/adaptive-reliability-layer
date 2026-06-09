from adaptive_reliability_layer.replay.buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
from adaptive_reliability_layer.replay.report import ReplayComparisonResult, StrategyReplaySummary


def test_compute_buyer_kpis_risk_first_narrative():
    result = ReplayComparisonResult(
        summaries=(
            StrategyReplaySummary(
                name="frozen",
                steps=10,
                mean_accuracy=0.96,
                mean_utility=0.90,
                mean_risk_capital=60.0,
                mean_shift_score=0.5,
                mean_reliability=0.88,
                intervention_rate=0.0,
                reset_count=0,
                risk_alert_count=8,
                first_drift_step=2,
                recommendation_rate=0.0,
                recommendation_execution_rate=0.0,
                bounded_interventions_per_1000=0.0,
                retrain_recommendation_count=4,
                first_retrain_recommendation_step=3,
                budget_limited_count=0,
            ),
            StrategyReplaySummary(
                name="bandit",
                steps=10,
                mean_accuracy=0.956,
                mean_utility=0.941,
                mean_risk_capital=4.0,
                mean_shift_score=0.4,
                mean_reliability=0.91,
                intervention_rate=0.2,
                reset_count=1,
                risk_alert_count=1,
                first_drift_step=3,
                recommendation_rate=0.3,
                recommendation_execution_rate=0.8,
                bounded_interventions_per_1000=200.0,
                retrain_recommendation_count=1,
                first_retrain_recommendation_step=8,
                budget_limited_count=0,
            ),
        ),
        controller_vs_frozen_utility_delta=0.041,
        controller_vs_frozen_risk_reduction=0.933,
        controller_vs_frozen_harmful_events_avoided=6,
        controller_vs_frozen_retrain_deferral_steps=5,
    )
    kpis = compute_buyer_kpis(result, controller_name="bandit")
    assert kpis is not None
    assert kpis.harmful_alert_reduction_pct >= 80.0
    assert kpis.risk_exposure_reduction_pct >= 90.0
    assert kpis.accuracy_equivalent is True
    report = render_buyer_replay_report(result, source_label="test")
    assert "Lead with risk" in report
    assert "Technical detail" in report
