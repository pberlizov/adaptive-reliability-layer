from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RegimeCoreset:
    """Compact recurrence-aware support substrate (DPCore / CTTA buffering line)."""

    max_size: int = 24
    features: np.ndarray | None = None
    labels: np.ndarray | None = None
    utility: np.ndarray | None = None
    reuse_quality: float = 0.5
    utility_ema: float = 0.5

    def update(
        self,
        *,
        batch_features: np.ndarray,
        batch_labels: np.ndarray,
        batch_utility: float,
    ) -> None:
        features = np.asarray(batch_features, dtype=np.float32)
        labels = np.asarray(batch_labels, dtype=np.int64).reshape(-1)
        if features.ndim != 2 or labels.shape[0] != features.shape[0]:
            return
        self.utility_ema = 0.85 * self.utility_ema + 0.15 * float(batch_utility)
        self.reuse_quality = float(min(1.0, max(0.0, 0.55 * self.reuse_quality + 0.45 * self.utility_ema)))
        candidate = self._diverse_subset(features, labels, max_rows=min(self.max_size, features.shape[0]))
        point_utility = np.full(candidate.shape[0], float(batch_utility), dtype=np.float32)
        if self.features is None:
            self.features = candidate
            self.labels = labels[: candidate.shape[0]].copy()
            self.utility = point_utility
            return
        merged_features = np.concatenate([self.features, candidate], axis=0)
        merged_labels = np.concatenate([self.labels, labels[: candidate.shape[0]]], axis=0)
        merged_utility = np.concatenate([self.utility, point_utility], axis=0)
        keep = self._select_diverse(
            merged_features,
            merged_labels,
            merged_utility,
            size=self.max_size,
        )
        self.features = merged_features[keep].astype(np.float32, copy=False)
        self.labels = merged_labels[keep].astype(np.int64, copy=False)
        self.utility = merged_utility[keep].astype(np.float32, copy=False)

    def support_features(self) -> np.ndarray | None:
        return None if self.features is None or self.features.size == 0 else self.features

    def mean_positive_rate(self, default: float = 0.5) -> float:
        if self.labels is None or self.labels.size == 0:
            return default
        return float(self.labels.mean())

    def _diverse_subset(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        max_rows: int,
    ) -> np.ndarray:
        if features.shape[0] <= max_rows:
            return features.copy()
        positive = np.flatnonzero(labels == 1)
        negative = np.flatnonzero(labels == 0)
        half = max(1, max_rows // 2)
        chosen: list[int] = []
        if positive.size > 0:
            chosen.extend(positive[: min(half, positive.size)].tolist())
        if negative.size > 0:
            chosen.extend(negative[: min(half, negative.size)].tolist())
        if len(chosen) < max_rows:
            remainder = [index for index in range(features.shape[0]) if index not in set(chosen)]
            chosen.extend(remainder[: max_rows - len(chosen)])
        return features[np.asarray(chosen[:max_rows], dtype=np.int64)].copy()

    def _select_diverse(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        utility: np.ndarray,
        *,
        size: int,
    ) -> np.ndarray:
        if features.shape[0] <= size:
            return np.arange(features.shape[0], dtype=np.int64)
        mean = features.mean(axis=0)
        scores = utility + 0.15 * np.linalg.norm(features - mean, axis=1)
        order = np.argsort(-scores)
        selected: list[int] = [int(order[0])]
        for index in order[1:]:
            if len(selected) >= size:
                break
            candidate = features[int(index)]
            distances = [float(np.linalg.norm(candidate - features[keep])) for keep in selected]
            if min(distances) > 1e-3 or labels[int(index)] != labels[selected[0]]:
                selected.append(int(index))
        if len(selected) < size:
            for index in order:
                if int(index) not in selected:
                    selected.append(int(index))
                if len(selected) >= size:
                    break
        return np.asarray(selected[:size], dtype=np.int64)


@dataclass
class ReservoirClusterRouter:
    """Online reservoir clustering on regime embeddings (ReservoirTTA line)."""

    max_clusters: int = 6
    centroids: list[np.ndarray] = field(default_factory=list)
    cluster_utility: list[float] = field(default_factory=list)
    cluster_counts: list[int] = field(default_factory=list)

    def assign(self, embedding: np.ndarray) -> int:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if not self.centroids:
            self.centroids.append(vector.copy())
            self.cluster_utility.append(0.5)
            self.cluster_counts.append(1)
            return 0
        distances = [float(np.linalg.norm(vector - centroid)) for centroid in self.centroids]
        best = int(np.argmin(distances))
        if distances[best] > 0.55 and len(self.centroids) < self.max_clusters:
            self.centroids.append(vector.copy())
            self.cluster_utility.append(0.5)
            self.cluster_counts.append(1)
            return len(self.centroids) - 1
        count = self.cluster_counts[best]
        self.centroids[best] = ((count * self.centroids[best]) + vector) / float(count + 1)
        self.cluster_counts[best] = count + 1
        return best

    def record_cluster_outcome(self, cluster_id: int, reward: float) -> None:
        if cluster_id < 0 or cluster_id >= len(self.cluster_utility):
            return
        previous = self.cluster_utility[cluster_id]
        self.cluster_utility[cluster_id] = 0.82 * previous + 0.18 * float(reward)

    def cluster_reuse_quality(self, cluster_id: int) -> float:
        if cluster_id < 0 or cluster_id >= len(self.cluster_utility):
            return 0.5
        return float(self.cluster_utility[cluster_id])
