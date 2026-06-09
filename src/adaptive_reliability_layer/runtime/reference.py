from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from ..tabular_benchmark import TabularBatch, TabularReferenceProfile, TabularShiftMonitor
from .model_adapter import ModelAdapter


def _binary_entropy(probability: float) -> float:
    epsilon = 1e-6
    clamped = min(max(probability, epsilon), 1.0 - epsilon)
    return -(clamped * math.log(clamped) + (1.0 - clamped) * math.log(1.0 - clamped))


def build_reference_profile_from_adapter(
    adapter: ModelAdapter,
    reference_batches: Iterable[TabularBatch],
) -> tuple[TabularReferenceProfile, list[float]]:
    """Build monitor reference statistics for any model adapter."""

    reference_batches = list(reference_batches)
    features = np.concatenate([batch.features for batch in reference_batches], axis=0)
    probabilities = [
        probability
        for batch in reference_batches
        for probability in adapter.predict_proba(batch.features)
    ]
    profile = TabularReferenceProfile(
        feature_mean=features.mean(axis=0),
        feature_variance=features.var(axis=0) + 1e-6,
        mean_entropy=float(np.mean([_binary_entropy(probability) for probability in probabilities])),
        mean_probability=float(np.mean(probabilities)),
        positive_rate=float(np.mean([1.0 if probability >= 0.5 else 0.0 for probability in probabilities])),
        mean_confidence=float(np.mean([max(probability, 1.0 - probability) for probability in probabilities])),
    )

    monitor = TabularShiftMonitor(profile)
    reference_scores: list[float] = []
    for batch in reference_batches:
        batch_probabilities = adapter.predict_proba(batch.features)
        signal = monitor.evaluate(batch.features, batch_probabilities)
        reference_scores.append(signal.output_score + 0.5 * signal.feature_score + signal.collapse_risk)
    return profile, reference_scores
