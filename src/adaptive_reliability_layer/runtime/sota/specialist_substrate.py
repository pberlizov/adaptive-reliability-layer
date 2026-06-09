from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SpecialistSubstrate:
    """Compact regime substrate (DPCore / buffering line) without recurrence gate."""

    coreset_size: int = 8
    support_vectors: list[np.ndarray] = field(default_factory=list)
    regime_descriptor: np.ndarray | None = None
    reuse_quality: float = 0.5
    utility_ema: float = 0.5

    def update_coreset(self, batch_signature: np.ndarray) -> None:
        vector = np.asarray(batch_signature, dtype=np.float32).reshape(-1)
        if not self.support_vectors:
            self.support_vectors.append(vector)
            return
        if len(self.support_vectors) < self.coreset_size:
            self.support_vectors.append(vector)
            return
        distances = [float(np.linalg.norm(vector - item)) for item in self.support_vectors]
        replace_index = int(np.argmin(distances))
        if distances[replace_index] < distances[-1] if len(distances) > 1 else 0.0:
            self.support_vectors[replace_index] = vector

    def record_outcome(self, reward: float) -> None:
        self.utility_ema = 0.85 * self.utility_ema + 0.15 * float(reward)
        self.reuse_quality = float(min(1.0, max(0.0, 0.6 * self.reuse_quality + 0.4 * self.utility_ema)))
