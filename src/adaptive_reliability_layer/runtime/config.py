from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .types import DEFAULT_BOUNDED_AUTO_ACTIONS, OperatingMode


@dataclass(frozen=True)
class MonitorConfig:
    alert_threshold: float = 1.1
    severe_threshold: float = 1.75
    risk_alert_threshold: float = 8.0
    risk_decay: float = 0.92


@dataclass(frozen=True)
class PolicyConfig:
    name: str = "multi_action"
    mild_threshold: float = 0.95
    severe_threshold: float = 1.55
    cooldown_steps: int = 2
    bandit_alpha: float = 0.75
    allowed_actions: tuple[str, ...] | None = None
    distance_threshold: float = 1.35
    max_specialists: int = 4
    scheduled_retrain_interval: int = 6
    threshold_learning_rate: float = 0.10
    use_behavior_signals: bool = True


@dataclass(frozen=True)
class GovernanceConfig:
    audit_db_path: str = ".arl/audit.db"
    snapshot_dir: str = ".arl/snapshots"
    max_snapshots: int = 200
    policy_version: str = "1.0.0"
    environment: str = "development"
    persist_snapshots_in_shadow: bool = False
    persist_snapshots_on_recommend: bool = False
    persist_snapshots_on_mutation: bool = True


@dataclass(frozen=True)
class MetricsConfig:
    enabled: bool = True
    prometheus_port: int = 9091
    namespace: str = "arl"


@dataclass(frozen=True)
class ReplayConfig:
    timestamp_column: str = "timestamp"
    label_column: str = "label"
    feature_prefix: str = "feature_"
    batch_size: int = 48
    label_delay_steps: int = 0
    label_delay_jitter_steps: int = 0
    max_steps: int | None = None


@dataclass(frozen=True)
class KpiConfigSpec:
    accuracy_weight: float = 1.0
    false_alert_cost: float = 0.06
    drift_cost: float = 0.03
    abstention_cost: float = 0.10
    reset_cost: float = 0.04
    retrain_recommendation_cost: float = 0.08


@dataclass(frozen=True)
class SafetyBudgetConfig:
    enabled: bool = True
    window_steps: int = 24
    max_auto_actions_per_window: int = 8
    max_resets_per_window: int = 1
    downgrade_to_recommend: bool = True
    rccda_loss_slope_block: bool = True


@dataclass(frozen=True)
class SotaExtensionsConfigSpec:
    enabled: bool = True
    asr_reset_enabled: bool = False
    online_conformal_enabled: bool = True
    target_coverage: float = 0.90
    timescale_enabled: bool = True
    drift_detector_enabled: bool = True
    proactive_drift_enabled: bool = False
    rccda_budget_enabled: bool = True
    deferred_adaptation_enabled: bool = False
    adaptation_safety_enabled: bool = True
    maintenance_latent_recenter: bool = True
    max_unsafe_adaptation_rate: float = 0.15


@dataclass(frozen=True)
class RuntimeConfig:
    operating_mode: OperatingMode = OperatingMode.SHADOW
    bounded_auto_actions: frozenset[str] = DEFAULT_BOUNDED_AUTO_ACTIONS
    model_version: str = "source-v1"
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    governance: GovernanceConfig = field(default_factory=GovernanceConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    safety_budget: SafetyBudgetConfig = field(default_factory=SafetyBudgetConfig)
    kpi: KpiConfigSpec = field(default_factory=KpiConfigSpec)
    policy_state_path: str | None = None
    policy_state_save_path: str | None = None
    policy_state_backend: str = "file"
    policy_state_redis_url: str | None = None
    policy_state_redis_key: str = "arl:policy:default"
    policy_state_encryption_key: str | None = None  # 32-byte hex AES-256 key for Redis backend
    sota: SotaExtensionsConfigSpec = field(default_factory=SotaExtensionsConfigSpec)
    log_json: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "RuntimeConfig":
        monitor = MonitorConfig(**data.get("monitor", {}))
        policy_data = dict(data.get("policy", {}))
        if policy_data.get("allowed_actions") is not None:
            policy_data["allowed_actions"] = tuple(policy_data["allowed_actions"])
        policy = PolicyConfig(**policy_data)
        governance = GovernanceConfig(**data.get("governance", {}))
        metrics = MetricsConfig(**data.get("metrics", {}))
        replay = ReplayConfig(**data.get("replay", {}))
        safety_budget = SafetyBudgetConfig(**data.get("safety_budget", {}))
        kpi = KpiConfigSpec(**data.get("kpi", {}))
        sota = SotaExtensionsConfigSpec(**data.get("sota", {}))
        mode = OperatingMode(data.get("operating_mode", OperatingMode.SHADOW.value))
        bounded = data.get("bounded_auto_actions")
        bounded_actions = (
            frozenset(bounded) if bounded is not None else DEFAULT_BOUNDED_AUTO_ACTIONS
        )
        return cls(
            operating_mode=mode,
            bounded_auto_actions=bounded_actions,
            model_version=str(data.get("model_version", "source-v1")),
            monitor=monitor,
            policy=policy,
            governance=governance,
            metrics=metrics,
            replay=replay,
            safety_budget=safety_budget,
            kpi=kpi,
            policy_state_path=data.get("policy_state_path"),
            policy_state_save_path=data.get("policy_state_save_path"),
            policy_state_backend=str(data.get("policy_state_backend", "file")),
            policy_state_redis_url=data.get("policy_state_redis_url"),
            policy_state_redis_key=str(data.get("policy_state_redis_key", "arl:policy:default")),
            sota=sota,
            log_json=bool(data.get("log_json", True)),
        )


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return RuntimeConfig.from_mapping(data)


def default_config_path() -> Path:
    from ..workspace import resolve_config_path

    return resolve_config_path("default.yaml")
