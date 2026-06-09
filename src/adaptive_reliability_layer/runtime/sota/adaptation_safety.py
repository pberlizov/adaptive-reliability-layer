from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AdaptationSafetyTracker:
    """Track guardrail-style safety under adaptation (TTT-undermines-guardrails line)."""

    window: int = 200
    _total_mutations: int = 0
    _unsafe_mutations: int = 0
    _high_risk_without_shadow: int = 0
    _collapse_after_adapt: int = 0
    _steps: list[int] = field(default_factory=list)

    def record(
        self,
        *,
        step: int,
        operating_mode: str,
        action_taken: str,
        collapse_risk: float,
        parameter_drift: float,
        force_shadow: bool,
        shift_score: float = 0.0,
    ) -> bool:
        mutated = action_taken not in {"none", "hold"}
        if not mutated:
            return True

        self._total_mutations += 1
        self._steps.append(step)
        while len(self._steps) > self.window:
            self._steps.pop(0)

        structural_shift = shift_score >= 1.75
        light_mutations = {"recalibrate", "label_shift", "bn_refresh", "hold"}
        bounded_mutations = light_mutations | {"adapt"}
        if structural_shift and action_taken in bounded_mutations and parameter_drift < 1.5:
            return True

        collapse_gate = 0.65 + 0.10 * min(1.0, max(0.0, shift_score - 1.2) / 1.2)
        drift_gate = 1.25 if not structural_shift else 1.85

        unsafe = False
        if operating_mode == "bounded_auto" and not force_shadow:
            if structural_shift:
                if action_taken == "reset" or (action_taken == "adapt" and parameter_drift >= 1.5):
                    self._high_risk_without_shadow += 1
                    unsafe = True
            elif collapse_risk >= collapse_gate:
                self._high_risk_without_shadow += 1
                unsafe = True
        if parameter_drift >= drift_gate and collapse_risk >= 0.45:
            if not (structural_shift and action_taken in bounded_mutations and parameter_drift < 1.35):
                self._collapse_after_adapt += 1
                unsafe = True
        if unsafe:
            self._unsafe_mutations += 1
        return not unsafe

    def summary(self) -> dict[str, float]:
        total = max(1, self._total_mutations)
        return {
            "mutation_count": float(self._total_mutations),
            "unsafe_mutation_rate": float(self._unsafe_mutations) / float(total),
            "high_risk_mutation_rate": float(self._high_risk_without_shadow) / float(total),
            "collapse_after_adapt_rate": float(self._collapse_after_adapt) / float(total),
            "adaptation_safety_ok": 1.0 if self._unsafe_mutations == 0 else 0.0,
        }

    def passes_deployment_gate(self, *, max_unsafe_rate: float = 0.15) -> bool:
        if self._total_mutations == 0:
            return True
        return (self._unsafe_mutations / self._total_mutations) <= max_unsafe_rate
