from __future__ import annotations

from dataclasses import dataclass, field

from ..runtime.layer import ReliabilityLayer
from ..runtime.types import DeploymentSurface


@dataclass
class ReplayRunState:
    name: str
    layer: ReliabilityLayer
    surfaces: list[DeploymentSurface] = field(default_factory=list)
    utilities: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    risk_capitals: list[float] = field(default_factory=list)
    shift_scores: list[float] = field(default_factory=list)
    revealed_metrics: list[dict[str, float | str | None]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
