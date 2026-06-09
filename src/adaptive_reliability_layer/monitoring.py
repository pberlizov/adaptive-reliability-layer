from __future__ import annotations

from dataclasses import dataclass
import math


def _binary_entropy(probability: float) -> float:
    epsilon = 1e-6
    clamped = min(max(probability, epsilon), 1.0 - epsilon)
    return -(clamped * math.log(clamped) + (1.0 - clamped) * math.log(1.0 - clamped))


@dataclass(frozen=True)
class ReferenceProfile:
    feature_mean: float
    feature_variance: float
    mean_entropy: float
    positive_rate: float
    mean_confidence: float


@dataclass(frozen=True)
class BatchStats:
    mean: float
    variance: float
    mean_entropy: float
    positive_rate: float
    mean_confidence: float


@dataclass(frozen=True)
class ShiftSignal:
    score: float
    feature_score: float
    output_score: float
    collapse_risk: float
    alert: bool
    severe: bool
    drift_direction: float
    stats: BatchStats


class ShiftMonitor:
    """Tracks simple batch statistics against a source reference."""

    def __init__(
        self,
        reference: ReferenceProfile,
        alert_threshold: float = 1.0,
        severe_threshold: float = 1.8,
        output_weight: float = 0.75,
        collapse_weight: float = 0.5,
    ) -> None:
        self._reference = reference
        self._alert_threshold = alert_threshold
        self._severe_threshold = severe_threshold
        self._output_weight = output_weight
        self._collapse_weight = collapse_weight

    def evaluate(self, features: list[float], probabilities: list[float]) -> ShiftSignal:
        mean = sum(features) / max(1, len(features))
        variance = sum((x - mean) ** 2 for x in features) / max(1, len(features))
        mean_entropy = sum(_binary_entropy(probability) for probability in probabilities) / max(1, len(probabilities))
        positive_rate = sum(int(probability >= 0.5) for probability in probabilities) / max(1, len(probabilities))
        mean_confidence = sum(max(probability, 1.0 - probability) for probability in probabilities) / max(
            1, len(probabilities)
        )

        mean_gap = abs(mean - self._reference.feature_mean)
        variance_gap = abs(variance - self._reference.feature_variance)
        feature_score = mean_gap + 0.5 * variance_gap

        entropy_gap = abs(mean_entropy - self._reference.mean_entropy)
        rate_gap = abs(positive_rate - self._reference.positive_rate)
        confidence_gap = abs(mean_confidence - self._reference.mean_confidence)
        output_score = entropy_gap + rate_gap + 0.5 * confidence_gap

        collapse_risk = max(0.0, self._reference.mean_entropy - mean_entropy) + max(
            0.0,
            abs(positive_rate - 0.5) - abs(self._reference.positive_rate - 0.5),
        )

        score = feature_score + self._output_weight * output_score + self._collapse_weight * collapse_risk
        return ShiftSignal(
            score=score,
            alert=score >= self._alert_threshold,
            severe=score >= self._severe_threshold or collapse_risk >= 0.35,
            feature_score=feature_score,
            output_score=output_score,
            collapse_risk=collapse_risk,
            drift_direction=mean - self._reference.feature_mean,
            stats=BatchStats(
                mean=mean,
                variance=variance,
                mean_entropy=mean_entropy,
                positive_rate=positive_rate,
                mean_confidence=mean_confidence,
            ),
        )


def build_reference_profile(features_batches: list[list[float]], probabilities_batches: list[list[float]]) -> ReferenceProfile:
    flat_features = [feature for batch in features_batches for feature in batch]
    mean = sum(flat_features) / max(1, len(flat_features))
    variance = sum((feature - mean) ** 2 for feature in flat_features) / max(1, len(flat_features))

    flat_probabilities = [probability for batch in probabilities_batches for probability in batch]
    mean_entropy = sum(_binary_entropy(probability) for probability in flat_probabilities) / max(1, len(flat_probabilities))
    positive_rate = sum(int(probability >= 0.5) for probability in flat_probabilities) / max(1, len(flat_probabilities))
    mean_confidence = sum(max(probability, 1.0 - probability) for probability in flat_probabilities) / max(
        1, len(flat_probabilities)
    )

    return ReferenceProfile(
        feature_mean=mean,
        feature_variance=variance,
        mean_entropy=mean_entropy,
        positive_rate=positive_rate,
        mean_confidence=mean_confidence,
    )
