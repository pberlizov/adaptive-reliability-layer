from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TimescaleExpert:
    name: str
    span: int
    ema_shift: float = 0.0
    ema_output: float = 0.0
    weight: float = 1.0


@dataclass
class MultiTimescaleController:
    """Black-box online shift adaptation via competing attention spans."""

    decay_short: float = 0.75
    decay_medium: float = 0.90
    decay_long: float = 0.97
    _experts: tuple[TimescaleExpert, ...] = field(
        default_factory=lambda: (
            TimescaleExpert("short", 4),
            TimescaleExpert("medium", 16),
            TimescaleExpert("long", 64),
        )
    )

    def update(self, *, shift_score: float, output_score: float) -> str:
        best_name = "medium"
        best_value = -1.0
        for expert in self._experts:
            if expert.span <= 8:
                decay = self.decay_short
            elif expert.span <= 24:
                decay = self.decay_medium
            else:
                decay = self.decay_long
            expert.ema_shift = decay * expert.ema_shift + (1.0 - decay) * shift_score
            expert.ema_output = decay * expert.ema_output + (1.0 - decay) * output_score
            urgency = 0.6 * expert.ema_shift + 0.4 * expert.ema_output
            expert.weight = max(0.05, urgency)
            if urgency > best_value:
                best_value = urgency
                best_name = expert.name
        return best_name

    def adaptation_gain(self, expert_name: str) -> float:
        lookup = {expert.name: expert for expert in self._experts}
        expert = lookup.get(expert_name, self._experts[1])
        return float(min(1.0, max(0.0, 0.5 * expert.ema_shift + 0.5 * expert.ema_output)))
