from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DeferredAdaptationJob:
    step: int
    action: str
    snapshot_before: object
    metadata: dict[str, Any]


@dataclass
class DeferredAdaptationQueue:
    """Caravan-style decoupling: score now, apply mutations on a slower path."""

    max_pending: int = 64
    _queue: list[DeferredAdaptationJob] = field(default_factory=list)

    def enqueue(self, job: DeferredAdaptationJob) -> bool:
        if len(self._queue) >= self.max_pending:
            return False
        self._queue.append(job)
        return True

    def flush(self) -> list[DeferredAdaptationJob]:
        jobs = list(self._queue)
        self._queue.clear()
        return jobs

    @property
    def pending_count(self) -> int:
        return len(self._queue)
