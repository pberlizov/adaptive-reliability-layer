from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class OnlineLogisticModel:
    """A tiny model used for early-stage simulation experiments."""

    weight: float = 1.0
    bias: float = 0.0
    source_weight: float = 1.0
    source_bias: float = 0.0

    def predict_proba(self, features: list[float]) -> list[float]:
        return [self._sigmoid(self.weight * x + self.bias) for x in features]

    def predict(self, features: list[float]) -> list[int]:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(features)]

    def adapt(
        self,
        features: list[float],
        probabilities: list[float],
        *,
        step_size: float,
        confidence_threshold: float,
        anchor_strength: float,
        max_parameter_drift: float,
    ) -> float:
        """Apply a bounded pseudo-label update and return the selected fraction."""

        selected_examples: list[tuple[float, float, int]] = []
        for feature, probability in zip(features, probabilities):
            confidence = max(probability, 1.0 - probability)
            if confidence < confidence_threshold:
                continue
            pseudo_label = 1 if probability >= 0.5 else 0
            selected_examples.append((feature, probability, pseudo_label))

        if not selected_examples:
            return 0.0

        grad_weight = 0.0
        grad_bias = 0.0
        for feature, probability, pseudo_label in selected_examples:
            error = probability - pseudo_label
            grad_weight += error * feature
            grad_bias += error

        scale = 1.0 / len(selected_examples)
        grad_weight *= scale
        grad_bias *= scale

        grad_weight += anchor_strength * (self.weight - self.source_weight)
        grad_bias += anchor_strength * (self.bias - self.source_bias)

        self.weight -= step_size * grad_weight
        self.bias -= step_size * grad_bias
        self.weight = self._clip_around_source(self.weight, self.source_weight, max_parameter_drift)
        self.bias = self._clip_around_source(self.bias, self.source_bias, max_parameter_drift)

        return len(selected_examples) / max(1, len(features))

    def clone(self) -> "OnlineLogisticModel":
        return OnlineLogisticModel(
            weight=self.weight,
            bias=self.bias,
            source_weight=self.source_weight,
            source_bias=self.source_bias,
        )

    def parameter_drift(self) -> float:
        return abs(self.weight - self.source_weight) + abs(self.bias - self.source_bias)

    def reset(self) -> None:
        self.weight = self.source_weight
        self.bias = self.source_bias

    @staticmethod
    def _sigmoid(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    @staticmethod
    def _clip_around_source(value: float, source_value: float, max_parameter_drift: float) -> float:
        lower_bound = source_value - max_parameter_drift
        upper_bound = source_value + max_parameter_drift
        return min(max(value, lower_bound), upper_bound)
