from __future__ import annotations

from adaptive_reliability_layer.replay.correction_path import (
    CorrectionWinCriteria,
    _evaluate_candidate,
)
from adaptive_reliability_layer.replay.failure_analysis import (
    DriverBucketSummary,
    ProductionFailureAnalysisReport,
    SourceFailureAnalysis,
    VariantAnalysis,
)


def _variant(
    *,
    name: str,
    utility: float,
    explicit: float = 0.0,
    burden: float = 0.01,
) -> VariantAnalysis:
    return VariantAnalysis(
        name=name,
        mean_accuracy=0.95,
        mean_utility=utility,
        recommendation_rate=0.0,
        recommendation_execution_rate=0.0,
        intervention_rate=0.0,
        explicit_action_rate=explicit,
        correction_applied_rate=1.0 if name != "frozen" else 0.0,
        correction_only_rate=1.0 if explicit == 0.0 and name != "frozen" else 0.0,
        mean_correction_delta=0.02,
        mean_correction_flips=1.0,
        revealed_alert_rate=0.10,
        revealed_burden=burden,
        revealed_event_avoids_vs_frozen=0,
        revealed_event_introductions_vs_frozen=0,
        mean_revealed_accuracy_delta_vs_frozen=0.01,
        adaptation_safety_rate=1.0,
        buckets={
            "correction_only": DriverBucketSummary(
                count=10,
                revealed_batches=10,
                mean_revealed_accuracy=0.95,
                mean_revealed_utility=utility,
                mean_correction_delta=0.02,
                mean_flips=1.0,
            )
        },
    )


def _source(source_id: str, *, candidate_utility: float, full_utility: float = 0.905, burden: float = 0.011) -> SourceFailureAnalysis:
    return SourceFailureAnalysis(
        source_id=source_id,
        description=f"{source_id} description",
        dataset_path=None,
        validation_accuracy=0.93,
        revealed_accuracy_alert_threshold=0.85,
        variants={
            "frozen": _variant(name="frozen", utility=0.850, burden=0.015),
            "scheduled_retrain": _variant(name="scheduled_retrain", utility=0.860, burden=0.014),
            "naive": _variant(name="naive", utility=0.840, burden=0.016),
            "full": _variant(name="full", utility=full_utility, explicit=0.02, burden=0.010),
            "correction_only": _variant(name="correction_only", utility=candidate_utility, explicit=0.0, burden=burden),
            "no_correction": _variant(name="no_correction", utility=0.884, explicit=0.02, burden=0.012),
            "no_explicit_actions": _variant(name="correction_plus_governor", utility=candidate_utility, explicit=0.0, burden=burden),
        },
        utility_driver="correction_dominant",
        full_vs_no_correction_utility_delta=0.021,
        full_vs_no_explicit_actions_utility_delta=0.001,
    )


def test_correction_candidate_passes_when_it_matches_full_closely():
    report = ProductionFailureAnalysisReport(
        config_path="config.yaml",
        controller_name="delayed_hybrid",
        baseline_strategies=("scheduled_retrain", "naive"),
        sources=(
            _source("a", candidate_utility=0.903, burden=0.011),
            _source("b", candidate_utility=0.902, burden=0.012),
            _source("c", candidate_utility=0.904, burden=0.010),
        ),
    )
    criteria = CorrectionWinCriteria()

    candidate = _evaluate_candidate(
        report=report,
        candidate_key="correction_only",
        label="correction_only",
        criteria=criteria,
    )

    assert candidate.passed is True
    assert candidate.sources_within_full_gap == 3
    assert candidate.mean_explicit_action_rate == 0.0


def test_correction_candidate_fails_when_it_loses_too_much_vs_full():
    report = ProductionFailureAnalysisReport(
        config_path="config.yaml",
        controller_name="delayed_hybrid",
        baseline_strategies=("scheduled_retrain", "naive"),
        sources=(
            _source("a", candidate_utility=0.892),
            _source("b", candidate_utility=0.893),
            _source("c", candidate_utility=0.894),
        ),
    )
    criteria = CorrectionWinCriteria()

    candidate = _evaluate_candidate(
        report=report,
        candidate_key="correction_only",
        label="correction_only",
        criteria=criteria,
    )

    assert candidate.passed is False
    failing = {name for name, passed, _ in candidate.checks if not passed}
    assert "mean_gap_vs_full" in failing or "worst_gap_vs_full" in failing
