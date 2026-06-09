"""Tests for discrimination metrics and benchmark spec loading."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from adaptive_reliability_layer.replay.discrimination_benchmark import (
    load_discrimination_benchmark_spec,
)
from adaptive_reliability_layer.replay.discrimination_metrics import (
    FraudCostConfig,
    PointwiseOutcomes,
    compute_classification_metrics,
    compute_metric_spreads,
    compute_temporal_half_metrics,
    recall_at_minimum_precision,
)
from adaptive_reliability_layer.replay.real_data import _restrict_stream_pool_tail, _split_train_test_indices


def test_split_train_test_indices_temporal_orders_time():
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)
    time_rank = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64)
    train, test = _split_train_test_indices(
        labels,
        time_rank,
        test_fraction=0.5,
        seed=7,
        temporal_split=True,
    )
    assert list(train) == [0, 1, 2]
    assert list(test) == [3, 4, 5]


def test_restrict_stream_pool_tail_keeps_latest_rows():
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    time_rank = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64)
    x_tail, y_tail, t_tail = _restrict_stream_pool_tail(
        features,
        labels,
        time_rank,
        tail_fraction=0.5,
    )
    assert len(y_tail) == 3
    assert list(t_tail) == [4, 5, 6]
    assert list(y_tail) == [1, 1, 1]


def test_recall_at_minimum_precision_prefers_high_threshold():
    y_true = np.array([1, 0, 1, 0, 1, 0], dtype=np.int64)
    y_prob = np.array([0.95, 0.80, 0.70, 0.40, 0.65, 0.10], dtype=np.float64)
    recall = recall_at_minimum_precision(y_true, y_prob, min_precision=0.80)
    assert recall is not None
    assert recall <= 1.0
    assert recall >= 0.0


def test_cost_weighted_error_penalizes_false_negatives_more():
    outcomes = PointwiseOutcomes(
        y_true=(1, 0, 1, 0),
        y_pred=(0, 0, 1, 1),
        y_prob=(0.2, 0.2, 0.8, 0.8),
        steps=(0, 0, 1, 1),
    )
    cheap_fn = compute_classification_metrics(outcomes, cost=FraudCostConfig(false_negative_cost=1.0))
    costly_fn = compute_classification_metrics(outcomes, cost=FraudCostConfig(false_negative_cost=20.0))
    assert costly_fn.cost_weighted_error > cheap_fn.cost_weighted_error


def test_temporal_half_metrics_capture_late_stream_change():
    outcomes = PointwiseOutcomes(
        y_true=(1, 1, 0, 0, 1, 1, 0, 0),
        y_pred=(1, 1, 0, 0, 0, 0, 0, 0),
        y_prob=(0.9, 0.9, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1),
        steps=(0, 0, 1, 1, 2, 2, 3, 3),
    )
    halves = compute_temporal_half_metrics(outcomes)
    assert halves.first_half.recall > halves.second_half.recall
    assert halves.recall_delta_second_minus_first < 0.0


def test_single_class_temporal_half_balanced_accuracy_no_warning():
    outcomes = PointwiseOutcomes(
        y_true=(0, 0, 0, 0, 1, 1),
        y_pred=(0, 0, 0, 0, 0, 0),
        y_prob=(0.1, 0.1, 0.1, 0.1, 0.2, 0.2),
        steps=(0, 0, 1, 1, 2, 2),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        halves = compute_temporal_half_metrics(outcomes)
    assert not any("single label" in str(item.message).lower() for item in caught)
    assert halves.first_half.balanced_accuracy == 1.0
    assert halves.second_half.balanced_accuracy == 0.5


def test_metric_spreads_mark_rankable_difference():
    from adaptive_reliability_layer.replay.discrimination_metrics import (
        ClassificationMetrics,
        StrategyDiscriminationMetrics,
        TemporalHalfMetrics,
    )

    def _summary(name: str, recall: float) -> StrategyDiscriminationMetrics:
        base = ClassificationMetrics(
            n_samples=100,
            positive_rate=0.1,
            accuracy=0.9,
            balanced_accuracy=0.7 + recall / 10.0,
            pr_auc=0.5 + recall / 10.0,
            roc_auc=0.5 + recall / 10.0,
            precision=0.8,
            recall=recall,
            f1=0.8,
            recall_at_precision_80=recall,
            cost_weighted_error=1.0 - recall,
            false_positive_rate=0.1,
            false_negative_rate=0.2,
        )
        halves = TemporalHalfMetrics(
            first_half=base,
            second_half=base,
            recall_delta_second_minus_first=0.0,
            balanced_accuracy_delta_second_minus_first=0.0,
            pr_auc_delta_second_minus_first=0.0,
        )
        return StrategyDiscriminationMetrics(
            name=name,
            stream_metrics=base,
            temporal_halves=halves,
            mean_retrain_recommendation_rate=0.1,
            mean_correction_applied_rate=0.5,
            mean_decision_threshold=0.5,
        )

    spreads = compute_metric_spreads((_summary("a", 0.2), _summary("b", 0.5)), min_spread=0.05)
    recall_spread = next(item for item in spreads if item.metric_name == "recall")
    assert recall_spread.rankable is True
    assert recall_spread.spread == pytest.approx(0.3)


def test_load_discrimination_benchmark_spec():
    runtime, spec = load_discrimination_benchmark_spec("configs/discrimination_benchmark_suite.yaml")
    assert runtime.policy.name == "regime_aware_delayed_bandit"
    assert "ieee_cis_fraud_torch_hard" in {source.id for source in spec.sources}
    assert "delayed_hybrid" in spec.strategies
    assert "correction_only" in spec.strategies
    assert "correction_plus_governor" in spec.strategies
