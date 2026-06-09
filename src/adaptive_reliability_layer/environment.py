from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class Batch:
    features: list[float]
    labels: list[int]
    regime: str


class SyntheticStream:
    """Produces a simple nonstationary binary-classification stream."""

    def __init__(self, seed: int = 7, batch_size: int = 64) -> None:
        self._rng = Random(seed)
        self._batch_size = batch_size
        self._step = 0

    def next_batch(self) -> Batch:
        regime = self._regime_for_step(self._step)
        self._step += 1

        if regime == "stable":
            center = 0.0
            noise = 0.6
        elif regime == "gradual_shift":
            center = min(2.0, 0.1 * (self._step - 20))
            noise = 0.8
        elif regime == "abrupt_shift":
            center = 2.5
            noise = 1.0
        else:
            center = 0.5
            noise = 0.7

        features: list[float] = []
        labels: list[int] = []
        for _ in range(self._batch_size):
            x = self._rng.gauss(center, noise)
            y = 1 if x + self._rng.gauss(0.0, 0.5) > 0.6 else 0
            features.append(x)
            labels.append(y)

        return Batch(features=features, labels=labels, regime=regime)

    def _regime_for_step(self, step: int) -> str:
        if step < 20:
            return "stable"
        if step < 40:
            return "gradual_shift"
        if step < 55:
            return "abrupt_shift"
        return "recurring_regime"


def build_stream(steps: int, seed: int = 7, batch_size: int = 64) -> list[Batch]:
    """Generate a reproducible stream so all baselines see the same data."""

    stream = SyntheticStream(seed=seed, batch_size=batch_size)
    return [stream.next_batch() for _ in range(steps)]
