from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..regime import RegimePrototype, StreamingRegimeEncoder


def export_policy_state(policy: Any) -> dict[str, Any]:
    """Serialize learnable controller state for restart-safe deployments."""

    if hasattr(policy, "_specialists") and hasattr(policy, "_regime_encoder") and not hasattr(policy, "_matrices"):
        specialists = []
        for slot in policy._specialists:  # type: ignore[attr-defined]
            specialists.append({
                "name": slot.name,
                "snapshot": _export_model_snapshot(slot.snapshot),
                "signature": slot.signature.tolist() if slot.signature is not None else None,
                "usage_count": int(slot.usage_count),
                "cumulative_reward": float(slot.cumulative_reward),
                "reward_ema": float(slot.reward_ema),
                "similarity_ema": float(slot.similarity_ema),
                "lift_ema": float(slot.lift_ema),
                "recurrence_reward_ema": float(slot.recurrence_reward_ema),
                "route_advantage_ema": float(slot.route_advantage_ema),
                "future_reuse_ema": float(slot.future_reuse_ema),
                "support_quality_ema": float(slot.support_quality_ema),
                "quality_ema": float(slot.quality_ema),
                "reveal_count": int(slot.reveal_count),
                "last_used_step": int(slot.last_used_step),
                "successful_reuses": int(slot.successful_reuses),
                "shadow_wins": int(slot.shadow_wins),
                "probation_remaining": int(slot.probation_remaining),
                "support_positive_rate": float(slot.support_positive_rate),
                "creation_positive_rate": float(slot.creation_positive_rate),
                "regime_anchor": slot.regime_anchor.tolist() if slot.regime_anchor is not None else None,
                "regime_confidence_ema": float(slot.regime_confidence_ema),
                "exchangeability_ema": float(slot.exchangeability_ema),
                "reservoir_cluster_id": int(slot.reservoir_cluster_id),
                "controller": export_policy_state(slot.controller),
            })
        return {
            "kind": "delayed_hybrid",
            "active_index": int(policy._active_index),  # type: ignore[attr-defined]
            "routing_step": int(policy._routing_step),  # type: ignore[attr-defined]
            "specialists": specialists,
            "encoder": _export_regime_encoder(policy._regime_encoder),  # type: ignore[attr-defined]
        }

    if hasattr(policy, "_regime_encoder") and hasattr(policy, "_matrices"):
        return {
            "kind": (
                "fraud_context_delayed_bandit"
                if getattr(policy, "_fraud_context_mode", False)
                else "fraud_rank_delayed_bandit"
                if getattr(policy, "_fraud_rank_mode", False)
                else "regime_aware_delayed_bandit"
            ),
            "actions": list(policy._actions),
            "matrices": {key: value.tolist() for key, value in policy._matrices.items()},
            "vectors": {key: value.tolist() for key, value in policy._vectors.items()},
            "residual_weights": policy._residual_weights.tolist(),
            "residual_bias": float(policy._residual_bias),
            "rank_weights": policy._rank_weights.tolist(),
            "rank_bias": float(policy._rank_bias),
            "segment_count": int(getattr(policy, "_segment_count", 0)),
            "residual_prototype_bias": {
                str(key): float(value) for key, value in policy._residual_prototype_bias.items()
            },
            "residual_prototype_weights": {
                str(key): value.tolist() for key, value in policy._residual_prototype_weights.items()
            },
            "residual_expert_weights": {
                str(key): value.tolist() for key, value in policy._residual_expert_weights.items()
            },
            "residual_expert_bias": {
                str(key): float(value) for key, value in policy._residual_expert_bias.items()
            },
            "threshold_bias": float(getattr(policy, "_threshold_bias", 0.0)),
            "threshold_prototype_bias": {
                str(key): float(value) for key, value in getattr(policy, "_threshold_prototype_bias", {}).items()
            },
            "threshold_expert_bias": {
                str(key): float(value) for key, value in getattr(policy, "_threshold_expert_bias", {}).items()
            },
            "residual_recent_bias": float(getattr(policy, "_residual_recent_bias", 0.0)),
            "residual_prototype_recent_bias": {
                str(key): float(value) for key, value in getattr(policy, "_residual_prototype_recent_bias", {}).items()
            },
            "encoder": _export_regime_encoder(policy._regime_encoder),
            "encoder_step": int(policy._encoder_step),
            "shift_ema": float(policy._shift_ema),
            "capital_ema": float(policy._capital_ema),
            "reliability_ema": float(policy._reliability_ema),
            "reward_ema": float(policy._reward_ema),
            "revealed_accuracy_ema": float(policy._revealed_accuracy_ema),
        }
    if hasattr(policy, "_matrices"):
        return {
            "kind": "bandit",
            "actions": list(policy._actions),
            "matrices": {key: value.tolist() for key, value in policy._matrices.items()},
            "vectors": {key: value.tolist() for key, value in policy._vectors.items()},
        }
    return {"kind": "static", "policy": policy.__class__.__name__}


def load_policy_state(policy: Any, state: dict[str, Any]) -> None:
    kind = state.get("kind", "static")
    if kind == "static":
        return
    if kind == "delayed_hybrid":
        if not hasattr(policy, "_specialists"):
            raise TypeError(f"policy {type(policy)!r} cannot load state kind={kind!r}")
        # New format: full specialist serialization including snapshots and metadata
        if "specialists" in state:
            from ..tabular_benchmark import SpecialistSlot
            policy._specialists = []  # type: ignore[attr-defined]
            for slot_data in state["specialists"]:
                snapshot = _load_model_snapshot(slot_data.get("snapshot"))
                sig_raw = slot_data.get("signature")
                signature = np.asarray(sig_raw, dtype=np.float64) if sig_raw is not None else np.zeros(28, dtype=np.float64)
                anchor_raw = slot_data.get("regime_anchor")
                regime_anchor = np.asarray(anchor_raw, dtype=np.float64) if anchor_raw is not None else None
                new_slot = SpecialistSlot(
                    name=str(slot_data.get("name", "specialist")),
                    snapshot=snapshot,
                    signature=signature,
                    controller=policy._new_specialist_controller(),  # type: ignore[attr-defined]
                    usage_count=int(slot_data.get("usage_count", 0)),
                    cumulative_reward=float(slot_data.get("cumulative_reward", 0.0)),
                    reward_ema=float(slot_data.get("reward_ema", 0.0)),
                    similarity_ema=float(slot_data.get("similarity_ema", 0.0)),
                    lift_ema=float(slot_data.get("lift_ema", 0.0)),
                    recurrence_reward_ema=float(slot_data.get("recurrence_reward_ema", 0.0)),
                    route_advantage_ema=float(slot_data.get("route_advantage_ema", 0.0)),
                    future_reuse_ema=float(slot_data.get("future_reuse_ema", 0.0)),
                    support_quality_ema=float(slot_data.get("support_quality_ema", 0.0)),
                    quality_ema=float(slot_data.get("quality_ema", 0.0)),
                    reveal_count=int(slot_data.get("reveal_count", 0)),
                    last_used_step=int(slot_data.get("last_used_step", 0)),
                    successful_reuses=int(slot_data.get("successful_reuses", 0)),
                    shadow_wins=int(slot_data.get("shadow_wins", 0)),
                    probation_remaining=int(slot_data.get("probation_remaining", 0)),
                    support_positive_rate=float(slot_data.get("support_positive_rate", 0.5)),
                    creation_positive_rate=float(slot_data.get("creation_positive_rate", 0.5)),
                    regime_anchor=regime_anchor,
                    regime_confidence_ema=float(slot_data.get("regime_confidence_ema", 0.0)),
                    exchangeability_ema=float(slot_data.get("exchangeability_ema", 0.0)),
                    reservoir_cluster_id=int(slot_data.get("reservoir_cluster_id", 0)),
                )
                if "controller" in slot_data:
                    load_policy_state(new_slot.controller, slot_data["controller"])
                policy._specialists.append(new_slot)  # type: ignore[attr-defined]
        # Legacy format: controller-only (pre-v0.3.3)
        elif "controllers" in state:
            controllers = state["controllers"]
            for specialist, controller_state in zip(policy._specialists, controllers):  # type: ignore[attr-defined]
                load_policy_state(specialist.controller, controller_state)
        policy._active_index = int(state.get("active_index", 0))  # type: ignore[attr-defined]
        policy._routing_step = int(state.get("routing_step", 0))  # type: ignore[attr-defined]
        if "encoder" in state:
            _load_regime_encoder(policy._regime_encoder, state["encoder"])  # type: ignore[attr-defined]
        return
    if not hasattr(policy, "_matrices"):
        raise TypeError(f"policy {type(policy)!r} cannot load state kind={kind!r}")

    actions = tuple(state["actions"])
    policy._actions = actions  # type: ignore[attr-defined]
    policy._matrices = {  # type: ignore[attr-defined]
        key: np.asarray(value, dtype=np.float64)
        for key, value in state["matrices"].items()
    }
    policy._vectors = {  # type: ignore[attr-defined]
        key: np.asarray(value, dtype=np.float64)
        for key, value in state["vectors"].items()
    }
    if kind in {"regime_aware_delayed_bandit", "fraud_rank_delayed_bandit", "fraud_context_delayed_bandit"}:
        policy._encoder_step = int(state.get("encoder_step", 0))  # type: ignore[attr-defined]
        policy._shift_ema = float(state.get("shift_ema", 0.0))  # type: ignore[attr-defined]
        policy._capital_ema = float(state.get("capital_ema", 0.0))  # type: ignore[attr-defined]
        policy._reliability_ema = float(state.get("reliability_ema", policy._reference.mean_confidence))  # type: ignore[attr-defined]
        policy._reward_ema = float(state.get("reward_ema", 0.0))  # type: ignore[attr-defined]
        policy._revealed_accuracy_ema = float(state.get("revealed_accuracy_ema", 0.5))  # type: ignore[attr-defined]
        if hasattr(policy, "_fraud_rank_mode"):
            policy._fraud_rank_mode = kind in {"fraud_rank_delayed_bandit", "fraud_context_delayed_bandit"}  # type: ignore[attr-defined]
        if hasattr(policy, "_fraud_context_mode"):
            policy._fraud_context_mode = kind == "fraud_context_delayed_bandit"  # type: ignore[attr-defined]
        if hasattr(policy, "_segment_count"):
            policy._segment_count = int(state.get("segment_count", getattr(policy, "_segment_count", 0)))  # type: ignore[attr-defined]
        policy._residual_weights = np.asarray(  # type: ignore[attr-defined]
            state.get("residual_weights", getattr(policy, "_residual_weights", [])),
            dtype=np.float64,
        )
        policy._residual_bias = float(state.get("residual_bias", getattr(policy, "_residual_bias", 0.0)))  # type: ignore[attr-defined]
        if hasattr(policy, "_rank_weights"):
            policy._rank_weights = np.asarray(  # type: ignore[attr-defined]
                state.get("rank_weights", getattr(policy, "_rank_weights", [])),
                dtype=np.float64,
            )
        if hasattr(policy, "_rank_bias"):
            policy._rank_bias = float(state.get("rank_bias", getattr(policy, "_rank_bias", 0.0)))  # type: ignore[attr-defined]
        policy._residual_prototype_bias = {  # type: ignore[attr-defined]
            int(key): float(value)
            for key, value in state.get("residual_prototype_bias", {}).items()
        }
        policy._residual_prototype_weights = {  # type: ignore[attr-defined]
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in state.get("residual_prototype_weights", {}).items()
        }
        if hasattr(policy, "_residual_expert_weights"):
            policy._residual_expert_weights = {  # type: ignore[attr-defined]
                str(key): np.asarray(value, dtype=np.float64)
                for key, value in state.get("residual_expert_weights", {}).items()
            } or getattr(policy, "_residual_expert_weights")
        if hasattr(policy, "_residual_expert_bias"):
            loaded_bias = {
                str(key): float(value)
                for key, value in state.get("residual_expert_bias", {}).items()
            }
            if loaded_bias:
                policy._residual_expert_bias.update(loaded_bias)  # type: ignore[attr-defined]
        if hasattr(policy, "_threshold_bias"):
            policy._threshold_bias = float(state.get("threshold_bias", 0.0))  # type: ignore[attr-defined]
        if hasattr(policy, "_threshold_prototype_bias"):
            policy._threshold_prototype_bias = {  # type: ignore[attr-defined]
                int(key): float(value)
                for key, value in state.get("threshold_prototype_bias", {}).items()
            }
        if hasattr(policy, "_threshold_expert_bias"):
            loaded_threshold_bias = {
                str(key): float(value)
                for key, value in state.get("threshold_expert_bias", {}).items()
            }
            if loaded_threshold_bias:
                policy._threshold_expert_bias.update(loaded_threshold_bias)  # type: ignore[attr-defined]
        policy._residual_recent_bias = float(state.get("residual_recent_bias", 0.0))  # type: ignore[attr-defined]
        policy._residual_prototype_recent_bias = {  # type: ignore[attr-defined]
            int(key): float(value)
            for key, value in state.get("residual_prototype_recent_bias", {}).items()
        }
        if "encoder" in state:
            _load_regime_encoder(policy._regime_encoder, state["encoder"])  # type: ignore[attr-defined]


def save_policy_state(policy: Any, path: str | Path) -> Path:
    return save_policy_state_atomic(policy, path)


def save_policy_state_atomic(policy: Any, path: str | Path) -> Path:
    import os

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(f"{output.suffix}.tmp")
    tmp.write_text(json.dumps(export_policy_state(policy), indent=2), encoding="utf-8")
    os.replace(tmp, output)
    return output


def load_policy_state_from_file(policy: Any, path: str | Path) -> None:
    state = json.loads(Path(path).read_text(encoding="utf-8"))
    load_policy_state(policy, state)


def _export_regime_encoder(encoder: StreamingRegimeEncoder) -> dict[str, Any]:
    prototypes = []
    for prototype in encoder._prototypes:  # noqa: SLF001 — persistence boundary
        prototypes.append(
            {
                "centroid": prototype.centroid.tolist(),
                "count": prototype.count,
                "reward_ema": prototype.reward_ema,
                "confidence_ema": prototype.confidence_ema,
                "novelty_ema": prototype.novelty_ema,
                "last_seen_step": prototype.last_seen_step,
            }
        )
    return {"prototypes": prototypes}


def _export_model_snapshot(snapshot: Any) -> dict[str, Any] | None:
    """Serialize a ModelSnapshot to a JSON-compatible dict."""
    if snapshot is None:
        return None
    try:
        return {
            "temperature": float(snapshot.temperature),
            "bias_offset": float(snapshot.bias_offset),
            "network_state": {
                key: tensor.detach().cpu().tolist()
                for key, tensor in snapshot.network_state.items()
            },
        }
    except Exception:
        return None


def _load_model_snapshot(data: dict[str, Any] | None) -> Any:
    """Deserialize a ModelSnapshot from a JSON-compatible dict."""
    if data is None:
        return None
    try:
        import torch

        from ..torch_model import ModelSnapshot

        return ModelSnapshot(
            network_state={
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in data["network_state"].items()
            },
            temperature=float(data.get("temperature", 1.0)),
            bias_offset=float(data.get("bias_offset", 0.0)),
        )
    except Exception:
        return None


def _load_regime_encoder(encoder: StreamingRegimeEncoder, payload: dict[str, Any]) -> None:
    encoder._prototypes = []  # noqa: SLF001
    for item in payload.get("prototypes", []):
        encoder._prototypes.append(  # noqa: SLF001
            RegimePrototype(
                centroid=np.asarray(item["centroid"], dtype=np.float64),
                count=int(item["count"]),
                reward_ema=float(item["reward_ema"]),
                confidence_ema=float(item["confidence_ema"]),
                novelty_ema=float(item["novelty_ema"]),
                last_seen_step=int(item["last_seen_step"]),
            )
        )
