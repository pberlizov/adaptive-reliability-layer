from __future__ import annotations

import math
from collections import Counter


def prediction_class_concentration(predictions: list[int]) -> float:
    """ASR-style collapse signal: fraction of batch predicted as the majority class."""

    if not predictions:
        return 0.0
    counts = Counter(predictions)
    majority = max(counts.values())
    return float(majority) / float(len(predictions))


def prediction_entropy_collapse(probabilities: list[float]) -> float:
    """Low entropy across confident wrong-ish batch → elevated collapse risk."""

    if not probabilities:
        return 0.0
    entropies = []
    for probability in probabilities:
        p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        entropies.append(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))
    mean_entropy = sum(entropies) / len(entropies)
    return float(max(0.0, 0.35 - mean_entropy))


def combined_asr_collapse_risk(
    predictions: list[int],
    probabilities: list[float],
    *,
    base_collapse_risk: float,
) -> tuple[float, float]:
    """Return (enhanced_collapse_risk, class_concentration)."""

    concentration = prediction_class_concentration(predictions)
    entropy_collapse = prediction_entropy_collapse(probabilities)
    asr_component = max(concentration - 0.5, 0.0) * 1.25 + entropy_collapse
    enhanced = float(max(base_collapse_risk, asr_component))
    return enhanced, concentration
