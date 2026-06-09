from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class DriftDetectorState:
    """CDSeer-inspired model-agnostic drift proxy without dense labeling."""

    window: int = 32
    alert_threshold: float = 0.55
    _positive_rates: deque[float] = field(default_factory=deque)
    _confidences: deque[float] = field(default_factory=deque)
    _output_scores: deque[float] = field(default_factory=deque)

    def observe(self, *, positive_rate: float, mean_confidence: float, output_score: float) -> float:
        self._positive_rates.append(positive_rate)
        self._confidences.append(mean_confidence)
        self._output_scores.append(output_score)
        while len(self._positive_rates) > self.window:
            self._positive_rates.popleft()
            self._confidences.popleft()
            self._output_scores.popleft()
        return self.score()

    def score(self) -> float:
        if len(self._positive_rates) < 8:
            return 0.0
        rates = np.asarray(self._positive_rates, dtype=np.float64)
        confidences = np.asarray(self._confidences, dtype=np.float64)
        outputs = np.asarray(self._output_scores, dtype=np.float64)
        rate_drift = float(np.std(rates) + abs(rates[-1] - np.median(rates)))
        confidence_drift = float(abs(confidences[-1] - np.mean(confidences)))
        output_trend = float(max(0.0, outputs[-1] - np.percentile(outputs, 25)))
        raw = 0.45 * rate_drift + 0.30 * confidence_drift + 0.25 * output_trend
        return float(min(1.0, raw))

    def should_intervene(self) -> bool:
        return self.score() >= self.alert_threshold

    def should_retrain(self) -> bool:
        return self.score() >= min(0.95, self.alert_threshold + 0.25)
