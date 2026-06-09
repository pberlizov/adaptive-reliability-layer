from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class RCCDABudgetGate:
    """Resource-aware update gate from loss/drift trend (Lyapunov-style heuristic)."""

    window: int = 16
    max_loss_slope: float = 0.08
    _proxy_losses: deque[float] = field(default_factory=deque)

    def observe_proxy_loss(self, *, shift_score: float, collapse_risk: float, miscoverage: float | None) -> float:
        proxy = shift_score + collapse_risk + (miscoverage or 0.0)
        self._proxy_losses.append(proxy)
        while len(self._proxy_losses) > self.window:
            self._proxy_losses.popleft()
        return self.loss_slope()

    def loss_slope(self) -> float:
        if len(self._proxy_losses) < 4:
            return 0.0
        values = list(self._proxy_losses)
        return float((values[-1] - values[0]) / max(1, len(values) - 1))

    def should_block_update(self) -> tuple[bool, str | None]:
        slope = self.loss_slope()
        if slope >= self.max_loss_slope:
            return True, "rccda_loss_slope"
        return False, None
