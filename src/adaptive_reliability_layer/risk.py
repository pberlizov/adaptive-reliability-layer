from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskState:
    raw_score: float
    p_value: float
    e_value: float
    capital: float
    alert: bool


class MartingaleRiskMonitor:
    """A lightweight sequential alarm based on a power-martingale style update."""

    def __init__(
        self,
        reference_scores: list[float],
        *,
        epsilon: float = 0.5,
        alert_threshold: float = 8.0,
        decay: float = 0.92,
        max_capital: float = 100.0,
    ) -> None:
        if not reference_scores:
            raise ValueError("reference_scores must not be empty")
        if not 0.0 < epsilon < 1.0:
            raise ValueError("epsilon must be in (0, 1)")

        self._reference_scores = sorted(reference_scores)
        self._epsilon = epsilon
        self._alert_threshold = alert_threshold
        self._decay = decay
        self._max_capital = max_capital
        self._capital = 1.0

    def update(self, raw_score: float) -> RiskState:
        tail_count = sum(score >= raw_score for score in self._reference_scores)
        p_value = (tail_count + 1.0) / (len(self._reference_scores) + 1.0)
        e_value = self._epsilon * (p_value ** (self._epsilon - 1.0))
        self._capital = min(self._max_capital, max(1.0, self._capital * self._decay * e_value))
        alert = self._capital >= self._alert_threshold
        return RiskState(
            raw_score=raw_score,
            p_value=p_value,
            e_value=e_value,
            capital=self._capital,
            alert=alert,
        )

    def reset(self) -> None:
        self._capital = 1.0

    def apply_mitigation(self, *, decay_factor: float = 0.55) -> RiskState:
        """Pull capital down after a successful hold/mitigation (controller-only path)."""
        if not 0.0 < decay_factor < 1.0:
            raise ValueError("decay_factor must be in (0, 1)")
        self._capital = max(1.0, 1.0 + (self._capital - 1.0) * decay_factor)
        alert = self._capital >= self._alert_threshold
        return RiskState(
            raw_score=0.0,
            p_value=1.0,
            e_value=1.0,
            capital=self._capital,
            alert=alert,
        )
