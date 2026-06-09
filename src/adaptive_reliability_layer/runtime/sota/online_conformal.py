from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OnlineConformalController:
    """Adaptive conformal inference (ACI-style) for the uncertainty lane."""

    target_coverage: float = 0.90
    learning_rate: float = 0.05
    alpha_min: float = 0.02
    alpha_max: float = 0.40
    _alpha: float = field(default=0.10, init=False)
    _scores: list[float] = field(default_factory=list, init=False)

    @property
    def alpha(self) -> float:
        return self._alpha

    def nonconformity(self, probability: float, label: int | None = None) -> float:
        p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
        if label is None:
            return 1.0 - max(p, 1.0 - p)
        predicted = 1 if p >= 0.5 else 0
        return 1.0 if predicted != int(label) else 1.0 - max(p, 1.0 - p)

    def interval_half_width(self, confidence: float) -> float:
        return float(self._alpha + (1.0 - confidence) * 0.5)

    def issue_action(
        self,
        *,
        mean_confidence: float,
        collapse_risk: float,
        shift_score: float = 0.0,
    ) -> str:
        width = self.interval_half_width(mean_confidence)
        structural_shift = shift_score >= 1.75
        collapse_tighten = 0.62 if structural_shift else 0.55
        width_tighten = 0.40 if structural_shift else 0.35
        if collapse_risk >= collapse_tighten or width >= width_tighten:
            return "tighten_abstention"
        if width <= 0.12 and not structural_shift:
            return "relax_abstention"
        if structural_shift and width <= 0.18 and collapse_risk < 0.50:
            return "relax_abstention"
        return "hold_threshold"

    def observe(self, score: float, *, hit: bool | None = None) -> dict[str, float]:
        self._scores.append(score)
        miscoverage = 0.0 if (hit is True) else 1.0
        if hit is None:
            miscoverage = 1.0 if score > self._alpha else 0.0
        target_miscoverage = 1.0 - self.target_coverage
        self._alpha = float(
            min(
                self.alpha_max,
                max(self.alpha_min, self._alpha + self.learning_rate * (miscoverage - target_miscoverage)),
            )
        )
        return {
            "alpha": self._alpha,
            "score": score,
            "miscoverage": miscoverage,
            "interval_half_width": self.interval_half_width(1.0 - score),
        }
