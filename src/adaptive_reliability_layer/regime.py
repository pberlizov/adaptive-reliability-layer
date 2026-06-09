from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return vector.astype(np.float64, copy=True)
    return (vector / norm).astype(np.float64, copy=False)


def build_regime_embedding(*components: np.ndarray) -> np.ndarray:
    """Build a compact regime embedding from heterogeneous stream signals."""

    normalized_components: list[np.ndarray] = []
    for component in components:
        flat = np.asarray(component, dtype=np.float64).reshape(-1)
        if flat.size == 0:
            continue
        scale = float(np.sqrt(np.mean(flat * flat))) + 1e-6
        normalized_components.append(np.tanh(flat / (2.5 * scale)))
    if not normalized_components:
        return np.zeros(1, dtype=np.float64)
    return _normalize(np.concatenate(normalized_components, axis=0))


@dataclass(frozen=True)
class ModelBehaviorSignature:
    """Compact model-behavior summary for regime detection."""

    mean_probability: float
    std_probability: float
    mean_entropy: float
    confidence_above_threshold: float  # fraction of predictions with max(p, 1-p) > 0.7


def compute_model_behavior_signature(
    probabilities: Union[list[float], np.ndarray],
) -> ModelBehaviorSignature:
    """Compute a compact model-behavior signature from a batch of predicted probabilities."""
    proba = np.asarray(probabilities, dtype=np.float64)
    entropy = -(proba * np.log(proba + 1e-9) + (1 - proba) * np.log(1 - proba + 1e-9))
    confidence = np.maximum(proba, 1 - proba)
    return ModelBehaviorSignature(
        mean_probability=float(np.mean(proba)),
        std_probability=float(np.std(proba)),
        mean_entropy=float(np.mean(entropy)),
        confidence_above_threshold=float(np.mean(confidence > 0.7)),
    )


def _behavior_feature_vector(sig: ModelBehaviorSignature) -> np.ndarray:
    """Return the behavior signal as a 4-element float64 vector."""
    return np.array(
        [sig.mean_probability, sig.std_probability, sig.mean_entropy, sig.confidence_above_threshold],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RegimeDescriptor:
    embedding: np.ndarray
    prototype_index: int
    similarity: float
    recurrence_confidence: float
    novelty_score: float
    prototype_reward: float
    prototype_count: int
    prototype_recency: float
    familiar: bool
    reuse_ready: bool
    prototype_positive_rate: float = 0.5  # EMA of revealed positive rates for this prototype


@dataclass
class RegimePrototype:
    centroid: np.ndarray
    count: int = 1
    reward_ema: float = 0.5
    confidence_ema: float = 0.35
    novelty_ema: float = 0.0
    last_seen_step: int = 0
    positive_rate_ema: float = 0.5  # revealed label positive rate associated with this prototype


class StreamingRegimeEncoder:
    """Prototype-based encoder for recurring nonstationary regimes."""

    def __init__(
        self,
        *,
        max_prototypes: int = 12,
        similarity_threshold: float = 0.93,
        familiarity_threshold: float = 0.58,
        reuse_threshold: float = 0.64,
        novelty_threshold: float = 0.28,
        creation_similarity_threshold: float = 0.88,
        staleness_horizon: int = 16,
        prototype_update_rate: float = 0.18,
        ema_decay: float = 0.84,
        use_behavior_signals: bool = True,
    ) -> None:
        self._max_prototypes = max_prototypes
        self._similarity_threshold = similarity_threshold
        self._familiarity_threshold = familiarity_threshold
        self._reuse_threshold = reuse_threshold
        self._novelty_threshold = novelty_threshold
        self._creation_similarity_threshold = creation_similarity_threshold
        self._staleness_horizon = staleness_horizon
        self._prototype_update_rate = prototype_update_rate
        self._ema_decay = ema_decay
        self._use_behavior_signals = use_behavior_signals
        self._prototypes: list[RegimePrototype] = []
        self._last_descriptor: RegimeDescriptor | None = None

    def _mixed_embedding(
        self,
        embedding: np.ndarray,
        probabilities: Union[list[float], np.ndarray, None],
    ) -> np.ndarray:
        """Return a mixed embedding that blends feature centroid with model-behavior signals.

        The blending is 40 % raw feature centroid (normalized) and 60 % model-behavior
        vector so that an operating-condition switch (which does NOT change model
        confidence or entropy) keeps the embeddings close, while true degradation
        (where confidence drops and entropy rises) creates a distinctly different
        embedding and is therefore classified separately.
        """
        if not self._use_behavior_signals:
            return embedding
        if probabilities is None:
            behavior = np.zeros(4, dtype=np.float64)
        else:
            sig = compute_model_behavior_signature(probabilities)
            behavior = _behavior_feature_vector(sig)
        # Scale behavior features to unit-ish magnitude comparable to the
        # already-normalized feature centroid.  The 4-element behavior vector
        # typically lives in [0, 1]; we scale it so its expected RMS matches
        # the feature centroid (norm ≈ 1 / sqrt(dim) per element).
        feature_dim = max(1, len(embedding))
        target_scale = 1.0 / float(np.sqrt(feature_dim))
        behavior_scale = float(np.sqrt(np.mean(behavior * behavior))) + 1e-6
        behavior_normalized = behavior * (target_scale / behavior_scale)
        # 40 % feature centroid, 60 % behavior — always same dimension (feature + 4)
        mixed = np.concatenate([0.40 * embedding, 0.60 * behavior_normalized], axis=0)
        return _normalize(mixed)

    def describe(
        self,
        embedding: np.ndarray,
        *,
        step: int,
        probabilities: Union[list[float], np.ndarray, None] = None,
    ) -> RegimeDescriptor:
        embedding = _normalize(np.asarray(embedding, dtype=np.float64).reshape(-1))
        embedding = self._mixed_embedding(embedding, probabilities)
        if self._prototypes and self._prototypes[0].centroid.shape != embedding.shape:
            self._prototypes.clear()
        if not self._prototypes:
            descriptor = RegimeDescriptor(
                embedding=embedding,
                prototype_index=-1,
                similarity=0.0,
                recurrence_confidence=0.0,
                novelty_score=1.0,
                prototype_reward=0.5,
                prototype_count=0,
                prototype_recency=0.0,
                familiar=False,
                reuse_ready=False,
            )
            self._last_descriptor = descriptor
            return descriptor

        similarities = [self._similarity(embedding, prototype.centroid) for prototype in self._prototypes]
        best_index = int(np.argmax(similarities))
        best_similarity = similarities[best_index]
        prototype = self._prototypes[best_index]
        prototype_recency = self._prototype_recency(step, prototype.last_seen_step)
        density = min(1.0, prototype.count / 4.0)
        reward_term = min(1.0, max(0.0, prototype.reward_ema - 0.45) / 0.20)
        confidence = float(
            np.clip(
                0.60 * best_similarity
                + 0.14 * density
                + 0.10 * reward_term
                + 0.10 * prototype_recency
                + 0.06 * prototype.confidence_ema,
                0.0,
                1.0,
            )
        )
        novelty = float(
            np.clip(
                max(0.0, 1.0 - best_similarity)
                + 0.18 * max(0.0, 0.60 - confidence)
                + 0.12 * prototype.novelty_ema,
                0.0,
                1.0,
            )
        )
        familiar = best_similarity >= self._similarity_threshold or confidence >= self._familiarity_threshold
        reuse_ready = familiar and confidence >= self._reuse_threshold and novelty <= self._novelty_threshold
        descriptor = RegimeDescriptor(
            embedding=embedding,
            prototype_index=best_index,
            similarity=best_similarity,
            recurrence_confidence=confidence,
            novelty_score=novelty,
            prototype_reward=prototype.reward_ema,
            prototype_count=prototype.count,
            prototype_recency=prototype_recency,
            familiar=familiar,
            reuse_ready=reuse_ready,
            prototype_positive_rate=prototype.positive_rate_ema,
        )
        self._last_descriptor = descriptor
        return descriptor

    def register(
        self,
        embedding: np.ndarray,
        *,
        step: int,
        probabilities: Union[list[float], np.ndarray, None] = None,
    ) -> RegimeDescriptor:
        descriptor = self.describe(embedding, step=step, probabilities=probabilities)
        if descriptor.prototype_index < 0 or descriptor.similarity < self._creation_similarity_threshold:
            index = self._allocate_prototype(descriptor, step=step)
            descriptor = self.describe(embedding, step=step, probabilities=probabilities)
            assert descriptor.prototype_index == index
        else:
            prototype = self._prototypes[descriptor.prototype_index]
            prototype.centroid = _normalize(
                (1.0 - self._prototype_update_rate) * prototype.centroid
                + self._prototype_update_rate * descriptor.embedding
            )
            prototype.count += 1
            prototype.confidence_ema = self._ema(prototype.confidence_ema, descriptor.recurrence_confidence)
            prototype.novelty_ema = self._ema(prototype.novelty_ema, descriptor.novelty_score)
            prototype.last_seen_step = step
            descriptor = self.describe(embedding, step=step, probabilities=probabilities)
        self._last_descriptor = descriptor
        return descriptor

    def reinforce(
        self,
        descriptor: RegimeDescriptor | None,
        *,
        step: int,
        reward: float,
        successful: bool,
        positive_rate: float | None = None,
    ) -> None:
        if descriptor is None:
            return
        if descriptor.prototype_index < 0 or descriptor.prototype_index >= len(self._prototypes):
            descriptor = self.register(descriptor.embedding, step=step)
        prototype = self._prototypes[descriptor.prototype_index]
        prototype.reward_ema = self._ema(prototype.reward_ema, reward)
        target_confidence = descriptor.recurrence_confidence + (0.10 if successful else -0.04)
        target_novelty = descriptor.novelty_score * (0.65 if successful else 1.10)
        prototype.confidence_ema = self._ema(prototype.confidence_ema, float(np.clip(target_confidence, 0.0, 1.0)))
        prototype.novelty_ema = self._ema(prototype.novelty_ema, float(np.clip(target_novelty, 0.0, 1.0)))
        prototype.last_seen_step = step
        if positive_rate is not None:
            prototype.positive_rate_ema = self._ema(prototype.positive_rate_ema, float(np.clip(positive_rate, 0.0, 1.0)))

    def get_diagnostics(self, *, prefix: str = "regime") -> dict[str, float]:
        if not self._prototypes:
            return {
                f"{prefix}_prototype_count": 0.0,
                f"{prefix}_mean_reward": 0.0,
                f"{prefix}_mean_confidence": 0.0,
                f"{prefix}_mean_novelty": 0.0,
                f"{prefix}_last_similarity": 0.0,
                f"{prefix}_last_confidence": 0.0,
                f"{prefix}_last_novelty": 0.0,
            }
        rewards = [prototype.reward_ema for prototype in self._prototypes]
        confidences = [prototype.confidence_ema for prototype in self._prototypes]
        novelties = [prototype.novelty_ema for prototype in self._prototypes]
        last_similarity = self._last_descriptor.similarity if self._last_descriptor is not None else 0.0
        last_confidence = self._last_descriptor.recurrence_confidence if self._last_descriptor is not None else 0.0
        last_novelty = self._last_descriptor.novelty_score if self._last_descriptor is not None else 0.0
        return {
            f"{prefix}_prototype_count": float(len(self._prototypes)),
            f"{prefix}_mean_reward": float(np.mean(rewards)),
            f"{prefix}_mean_confidence": float(np.mean(confidences)),
            f"{prefix}_mean_novelty": float(np.mean(novelties)),
            f"{prefix}_last_similarity": last_similarity,
            f"{prefix}_last_confidence": last_confidence,
            f"{prefix}_last_novelty": last_novelty,
        }

    def _allocate_prototype(self, descriptor: RegimeDescriptor, *, step: int) -> int:
        new_prototype = RegimePrototype(
            centroid=descriptor.embedding.copy(),
            count=1,
            reward_ema=max(0.45, descriptor.prototype_reward),
            confidence_ema=max(0.30, descriptor.recurrence_confidence),
            novelty_ema=descriptor.novelty_score,
            last_seen_step=step,
        )
        if len(self._prototypes) < self._max_prototypes:
            self._prototypes.append(new_prototype)
            return len(self._prototypes) - 1

        weakest_index = self._weakest_prototype_index(step=step)
        self._prototypes[weakest_index] = new_prototype
        return weakest_index

    def _weakest_prototype_index(self, *, step: int) -> int:
        weakest_score = float("inf")
        weakest_index = 0
        for index, prototype in enumerate(self._prototypes):
            recency = self._prototype_recency(step, prototype.last_seen_step)
            score = (
                prototype.reward_ema
                + 0.18 * prototype.confidence_ema
                + 0.10 * min(1.0, prototype.count / 4.0)
                + 0.06 * recency
                - 0.10 * prototype.novelty_ema
            )
            if score < weakest_score:
                weakest_score = score
                weakest_index = index
        return weakest_index

    def _similarity(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        return float(np.dot(lhs, rhs))

    def _prototype_recency(self, step: int, last_seen_step: int) -> float:
        age = max(0, step - last_seen_step)
        return max(0.0, 1.0 - age / max(1, self._staleness_horizon))

    def _ema(self, previous: float, new_value: float) -> float:
        return self._ema_decay * previous + (1.0 - self._ema_decay) * new_value
