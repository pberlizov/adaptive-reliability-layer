from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from statistics import mean

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

from ..runtime.types import DeploymentSurface
from .loader import ReplayStream, iter_replay_batches
from .types import ReplayRunState


@dataclass(frozen=True)
class FraudCostConfig:
    """Asymmetric fraud review / chargeback costs (relative units)."""

    false_positive_cost: float = 1.0
    false_negative_cost: float = 10.0


@dataclass(frozen=True)
class PointwiseOutcomes:
    y_true: tuple[int, ...]
    y_pred: tuple[int, ...]
    y_prob: tuple[float, ...]
    steps: tuple[int, ...]


@dataclass(frozen=True)
class ClassificationMetrics:
    n_samples: int
    positive_rate: float
    accuracy: float
    balanced_accuracy: float
    pr_auc: float | None
    roc_auc: float | None
    precision: float
    recall: float
    f1: float
    recall_at_precision_80: float | None
    cost_weighted_error: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass(frozen=True)
class TemporalHalfMetrics:
    first_half: ClassificationMetrics
    second_half: ClassificationMetrics
    recall_delta_second_minus_first: float
    balanced_accuracy_delta_second_minus_first: float
    pr_auc_delta_second_minus_first: float | None


@dataclass(frozen=True)
class StrategyDiscriminationMetrics:
    name: str
    stream_metrics: ClassificationMetrics
    temporal_halves: TemporalHalfMetrics
    mean_retrain_recommendation_rate: float
    mean_correction_applied_rate: float
    mean_decision_threshold: float


@dataclass(frozen=True)
class MetricSpread:
    metric_name: str
    values_by_strategy: dict[str, float | None]
    spread: float | None
    rankable: bool


def extract_pointwise_outcomes(
    stream: ReplayStream,
    run: ReplayRunState,
    *,
    batch_size: int,
    max_steps: int | None = None,
) -> PointwiseOutcomes:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    steps: list[int] = []
    for step_index, (_step, batch, _delayed) in enumerate(
        iter_replay_batches(stream, batch_size=batch_size, max_steps=max_steps)
    ):
        if step_index >= len(run.surfaces):
            break
        if batch.labels is None:
            continue
        labels = np.asarray(batch.labels, dtype=np.int64)
        surface = run.surfaces[step_index]
        predictions = np.asarray(surface.predictions, dtype=np.int64)
        probabilities = np.asarray(surface.probabilities, dtype=np.float64)
        if len(predictions) != len(labels):
            limit = min(len(predictions), len(labels))
            labels = labels[:limit]
            predictions = predictions[:limit]
            probabilities = probabilities[:limit]
        for index in range(len(labels)):
            y_true.append(int(labels[index]))
            y_pred.append(int(predictions[index]))
            y_prob.append(float(probabilities[index]))
            steps.append(step_index)
    return PointwiseOutcomes(
        y_true=tuple(y_true),
        y_pred=tuple(y_pred),
        y_prob=tuple(y_prob),
        steps=tuple(steps),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def _balanced_accuracy_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Balanced accuracy for binary fraud labels without sklearn shape warnings."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class = np.diag(cm) / cm.sum(axis=1)
    if np.any(np.isnan(per_class)):
        per_class = per_class[~np.isnan(per_class)]
    if len(per_class) == 0:
        return 0.0
    return float(np.mean(per_class))


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision + recall <= 0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def recall_at_minimum_precision(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_precision: float,
) -> float | None:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    prob_sorted = y_prob[order]
    best_recall = 0.0
    found = False
    tp = 0.0
    fp = 0.0
    positives = float(np.sum(y_true == 1))
    for index in range(len(y_sorted)):
        if y_sorted[index] == 1:
            tp += 1.0
        else:
            fp += 1.0
        precision = _safe_div(tp, tp + fp)
        if precision >= min_precision:
            recall = _safe_div(tp, positives)
            best_recall = max(best_recall, recall)
            found = True
        elif found and prob_sorted[index] < prob_sorted[index - 1]:
            break
    return best_recall if found else 0.0


def compute_classification_metrics(
    outcomes: PointwiseOutcomes,
    *,
    cost: FraudCostConfig = FraudCostConfig(),
) -> ClassificationMetrics:
    if not outcomes.y_true:
        return ClassificationMetrics(
            n_samples=0,
            positive_rate=0.0,
            accuracy=0.0,
            balanced_accuracy=0.0,
            pr_auc=None,
            roc_auc=None,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            recall_at_precision_80=None,
            cost_weighted_error=0.0,
            false_positive_rate=0.0,
            false_negative_rate=0.0,
        )
    y_true = np.asarray(outcomes.y_true, dtype=np.int64)
    y_pred = np.asarray(outcomes.y_pred, dtype=np.int64)
    y_prob = np.asarray(outcomes.y_prob, dtype=np.float64)
    positives = float(np.sum(y_true == 1))
    negatives = float(np.sum(y_true == 0))
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    precision, recall, f1 = _precision_recall_f1(y_true, y_pred)
    pr_auc = None
    roc_auc = None
    if len(np.unique(y_true)) >= 2:
        pr_auc = float(average_precision_score(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))
    total_cost = (
        cost.false_positive_cost * fp + cost.false_negative_cost * fn
    ) / max(1.0, len(y_true))
    return ClassificationMetrics(
        n_samples=len(y_true),
        positive_rate=float(np.mean(y_true)),
        accuracy=float(np.mean(y_true == y_pred)),
        balanced_accuracy=_balanced_accuracy_binary(y_true, y_pred),
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        precision=precision,
        recall=recall,
        f1=f1,
        recall_at_precision_80=recall_at_minimum_precision(
            y_true,
            y_prob,
            min_precision=0.80,
        ),
        cost_weighted_error=total_cost,
        false_positive_rate=_safe_div(fp, negatives),
        false_negative_rate=_safe_div(fn, positives),
    )


def _slice_outcomes(outcomes: PointwiseOutcomes, *, mask: np.ndarray) -> PointwiseOutcomes:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return PointwiseOutcomes(y_true=(), y_pred=(), y_prob=(), steps=())
    return PointwiseOutcomes(
        y_true=tuple(outcomes.y_true[index] for index in indices),
        y_pred=tuple(outcomes.y_pred[index] for index in indices),
        y_prob=tuple(outcomes.y_prob[index] for index in indices),
        steps=tuple(outcomes.steps[index] for index in indices),
    )


def compute_temporal_half_metrics(
    outcomes: PointwiseOutcomes,
    *,
    cost: FraudCostConfig = FraudCostConfig(),
) -> TemporalHalfMetrics:
    if not outcomes.steps:
        empty = compute_classification_metrics(PointwiseOutcomes((), (), (), ()), cost=cost)
        return TemporalHalfMetrics(
            first_half=empty,
            second_half=empty,
            recall_delta_second_minus_first=0.0,
            balanced_accuracy_delta_second_minus_first=0.0,
            pr_auc_delta_second_minus_first=None,
        )
    steps = np.asarray(outcomes.steps, dtype=np.int64)
    midpoint = (int(steps.min()) + int(steps.max()) + 1) // 2
    first = _slice_outcomes(outcomes, mask=steps < midpoint)
    second = _slice_outcomes(outcomes, mask=steps >= midpoint)
    first_metrics = compute_classification_metrics(first, cost=cost)
    second_metrics = compute_classification_metrics(second, cost=cost)
    pr_delta = None
    if first_metrics.pr_auc is not None and second_metrics.pr_auc is not None:
        pr_delta = second_metrics.pr_auc - first_metrics.pr_auc
    return TemporalHalfMetrics(
        first_half=first_metrics,
        second_half=second_metrics,
        recall_delta_second_minus_first=second_metrics.recall - first_metrics.recall,
        balanced_accuracy_delta_second_minus_first=(
            second_metrics.balanced_accuracy - first_metrics.balanced_accuracy
        ),
        pr_auc_delta_second_minus_first=pr_delta,
    )


def summarize_strategy_discrimination(
    name: str,
    stream: ReplayStream,
    run: ReplayRunState,
    *,
    batch_size: int,
    max_steps: int | None = None,
    cost: FraudCostConfig = FraudCostConfig(),
) -> StrategyDiscriminationMetrics:
    outcomes = extract_pointwise_outcomes(
        stream,
        run,
        batch_size=batch_size,
        max_steps=max_steps,
    )
    stream_metrics = compute_classification_metrics(outcomes, cost=cost)
    temporal_halves = compute_temporal_half_metrics(outcomes, cost=cost)
    surfaces = run.surfaces
    retrain_rate = sum(surface.retrain_recommended for surface in surfaces) / max(1, len(surfaces))
    correction_rate = sum(surface.correction_applied for surface in surfaces) / max(1, len(surfaces))
    threshold_mean = mean(surface.decision_threshold for surface in surfaces) if surfaces else 0.5
    return StrategyDiscriminationMetrics(
        name=name,
        stream_metrics=stream_metrics,
        temporal_halves=temporal_halves,
        mean_retrain_recommendation_rate=retrain_rate,
        mean_correction_applied_rate=correction_rate,
        mean_decision_threshold=threshold_mean,
    )


def compute_metric_spreads(
    summaries: tuple[StrategyDiscriminationMetrics, ...],
    *,
    min_spread: float = 0.005,
) -> tuple[MetricSpread, ...]:
    metric_extractors: tuple[tuple[str, Callable[[StrategyDiscriminationMetrics], float | None]], ...] = (
        ("balanced_accuracy", lambda item: item.stream_metrics.balanced_accuracy),
        ("pr_auc", lambda item: item.stream_metrics.pr_auc),
        ("recall", lambda item: item.stream_metrics.recall),
        ("recall_at_precision_80", lambda item: item.stream_metrics.recall_at_precision_80),
        ("cost_weighted_error", lambda item: item.stream_metrics.cost_weighted_error),
        ("f1", lambda item: item.stream_metrics.f1),
        ("second_half_recall", lambda item: item.temporal_halves.second_half.recall),
        ("recall_delta_late_minus_early", lambda item: item.temporal_halves.recall_delta_second_minus_first),
        ("retrain_recommendation_rate", lambda item: item.mean_retrain_recommendation_rate),
    )
    spreads: list[MetricSpread] = []
    for metric_name, extractor in metric_extractors:
        values: dict[str, float | None] = {}
        numeric: list[float] = []
        for summary in summaries:
            raw = extractor(summary)
            values[summary.name] = raw
            if raw is not None:
                numeric.append(float(raw))
        spread = None
        rankable = False
        if len(numeric) >= 2:
            spread = max(numeric) - min(numeric)
            rankable = spread >= min_spread
        spreads.append(
            MetricSpread(
                metric_name=metric_name,
                values_by_strategy=values,
                spread=spread,
                rankable=rankable,
            )
        )
    return tuple(spreads)


def benchmark_has_headroom(
    frozen: StrategyDiscriminationMetrics,
    *,
    max_balanced_accuracy: float = 0.95,
    min_positive_rate: float = 0.001,
) -> tuple[bool, str]:
    metrics = frozen.stream_metrics
    reasons: list[str] = []
    if metrics.n_samples <= 0:
        return False, "no labeled stream samples"
    if metrics.positive_rate < min_positive_rate:
        reasons.append(f"extreme imbalance (positive rate {metrics.positive_rate:.4f})")
    if metrics.balanced_accuracy >= max_balanced_accuracy:
        reasons.append(
            f"frozen balanced accuracy already high ({metrics.balanced_accuracy:.3f})"
        )
    late_recall = frozen.temporal_halves.second_half.recall
    early_recall = frozen.temporal_halves.first_half.recall
    if late_recall >= early_recall - 0.01:
        reasons.append("limited late-stream recall degradation on frozen baseline")
    if reasons:
        return False, "; ".join(reasons)
    return True, "frozen baseline shows measurable headroom on imbalance-aware metrics"
