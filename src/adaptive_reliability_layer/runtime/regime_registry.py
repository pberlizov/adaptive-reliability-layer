"""Cross-model regime knowledge sharing registry.

Allows multiple ARL sidecars (or multiple models within the same sidecar)
to share learned regime prototypes — so that a fraud model that already
learned "holiday spike = operating-condition switch" can provide a warm-start
for a newly deployed credit model on the same feature schema.

Architecture
------------
The registry is a thin key-value store keyed by ``(model_family, schema_hash)``.
Each value is a list of serialized ``RegimePrototype`` dicts.

Two backends are supported:
  - ``InMemoryRegimeRegistry``   — single-process; useful for multi-model serving
  - ``RedisRegimeRegistry``      — multi-process / multi-host; requires redis package

Privacy note: regime centroids are computed from customer feature distributions.
Before sharing across tenants, verify that centroid disclosure is permissible
under your data-use agreement.  Use the same-tenant guard (``model_family`` key)
to prevent cross-tenant leakage.

Usage
-----
    from adaptive_reliability_layer.runtime.regime_registry import (
        InMemoryRegimeRegistry, push_regime_state, pull_regime_state
    )

    registry = InMemoryRegimeRegistry()

    # After model A has learned regimes:
    push_regime_state(registry, policy=layer_a._policy, model_family="fraud_v2")

    # When model B starts fresh:
    pull_regime_state(registry, policy=layer_b._policy, model_family="fraud_v2")
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RegimeRegistry(Protocol):
    """Key-value store for serialized regime prototype lists."""

    def put(self, key: str, prototypes: list[dict[str, Any]]) -> None: ...
    def get(self, key: str) -> list[dict[str, Any]] | None: ...
    def keys(self) -> list[str]: ...


class InMemoryRegimeRegistry:
    """Thread-safe in-process regime registry for single-host multi-model setups."""

    def __init__(self) -> None:
        import threading
        self._store: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def put(self, key: str, prototypes: list[dict[str, Any]]) -> None:
        with self._lock:
            self._store[key] = [dict(p) for p in prototypes]

    def get(self, key: str) -> list[dict[str, Any]] | None:
        with self._lock:
            return [dict(p) for p in self._store[key]] if key in self._store else None

    def keys(self) -> list[str]:
        with self._lock:
            return list(self._store)


class RedisRegimeRegistry:
    """Redis-backed regime registry for multi-host sidecar pools.

    Requires: pip install redis
    """

    def __init__(self, url: str, key_prefix: str = "arl:regimes:", ttl_seconds: int = 86400) -> None:
        self._url = url
        self._prefix = key_prefix
        self._ttl = ttl_seconds

    def _client(self):
        try:
            import redis
        except ImportError as exc:
            raise ImportError("redis required for RedisRegimeRegistry") from exc
        return redis.from_url(self._url, decode_responses=True)

    def put(self, key: str, prototypes: list[dict[str, Any]]) -> None:
        client = self._client()
        full_key = f"{self._prefix}{key}"
        payload = json.dumps(prototypes)
        client.set(full_key, payload, ex=self._ttl)

    def get(self, key: str) -> list[dict[str, Any]] | None:
        raw = self._client().get(f"{self._prefix}{key}")
        if raw is None:
            return None
        return json.loads(raw)

    def keys(self) -> list[str]:
        prefix = self._prefix
        return [k[len(prefix):] for k in self._client().scan_iter(f"{prefix}*")]


# ---------------------------------------------------------------------------
# Push / pull helpers
# ---------------------------------------------------------------------------

def _schema_hash(feature_dim: int) -> str:
    """Simple hash to identify compatible feature schemas."""
    return hashlib.sha256(f"dim={feature_dim}".encode()).hexdigest()[:8]


def push_regime_state(
    registry: RegimeRegistry,
    policy: Any,
    *,
    model_family: str,
    max_prototypes: int = 20,
) -> int:
    """Push the policy's regime encoder prototypes to the registry.

    Returns the number of prototypes pushed.
    """
    from ..runtime.policy_state import _export_regime_encoder

    encoder = getattr(policy, "_regime_encoder", None)
    if encoder is None:
        # Try inner controller of hybrid policy
        specialists = getattr(policy, "_specialists", [])
        if specialists:
            encoder = getattr(specialists[0].controller, "_regime_encoder", None)
    if encoder is None:
        return 0

    prototypes_data = _export_regime_encoder(encoder)["prototypes"]
    if not prototypes_data:
        return 0

    # Keep only the top-N by reward_ema to avoid filling registry with noise
    top = sorted(prototypes_data, key=lambda p: -p.get("reward_ema", 0.0))[:max_prototypes]
    feature_dim = len(top[0]["centroid"]) if top else 0
    key = f"{model_family}:{_schema_hash(feature_dim)}"
    registry.put(key, top)
    return len(top)


def pull_regime_state(
    registry: RegimeRegistry,
    policy: Any,
    *,
    model_family: str,
    merge: bool = True,
) -> int:
    """Pull regime prototypes from the registry into the policy's encoder.

    When ``merge=True`` (default), existing prototypes are kept and the
    registry's prototypes are added — useful for warm-starting a new sidecar
    without discarding locally learned regimes.

    Returns the number of prototypes loaded.
    """
    from ..runtime.policy_state import _export_regime_encoder, _load_regime_encoder

    encoder = getattr(policy, "_regime_encoder", None)
    if encoder is None:
        return 0

    existing_data = _export_regime_encoder(encoder)["prototypes"]
    feature_dim = len(existing_data[0]["centroid"]) if existing_data else 0

    # Try to find a compatible key
    matching_key: str | None = None
    for key in registry.keys():
        if key.startswith(f"{model_family}:"):
            matching_key = key
            break
    if matching_key is None:
        return 0

    incoming = registry.get(matching_key) or []
    if not incoming:
        return 0

    if merge and existing_data:
        # Deduplicate by approximate centroid similarity
        merged = list(existing_data)
        for incoming_proto in incoming:
            merged.append(incoming_proto)
        _load_regime_encoder(encoder, {"prototypes": merged})
    else:
        _load_regime_encoder(encoder, {"prototypes": incoming})

    return len(incoming)
