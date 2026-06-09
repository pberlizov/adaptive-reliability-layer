from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class ProactiveDriftMonitor:
    """Proceed-inspired trend detector: adapt before performance cliff."""

    horizon: int = 12
    trigger_slope: float = 0.04
    _shift_trace: deque[float] = field(default_factory=deque)

    def observe(self, shift_score: float) -> tuple[bool, float]:
        self._shift_trace.append(shift_score)
        while len(self._shift_trace) > self.horizon:
            self._shift_trace.popleft()
        if len(self._shift_trace) < 4:
            return False, 0.0
        values = list(self._shift_trace)
        slope = (values[-1] - values[0]) / max(1, len(values) - 1)
        return slope >= self.trigger_slope, float(slope)
