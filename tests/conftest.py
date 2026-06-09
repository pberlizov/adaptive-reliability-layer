from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_reliability_layer.runtime.config import RuntimeConfig, load_runtime_config
from adaptive_reliability_layer.runtime.types import OperatingMode


@pytest.fixture
def temp_governance_dirs(tmp_path: Path):
    audit_db = tmp_path / "audit.db"
    snapshot_dir = tmp_path / "snapshots"
    return audit_db, snapshot_dir


@pytest.fixture
def runtime_config(temp_governance_dirs) -> RuntimeConfig:
    audit_db, snapshot_dir = temp_governance_dirs
    base = load_runtime_config("configs/default.yaml")
    return RuntimeConfig(
        operating_mode=OperatingMode.SHADOW,
        bounded_auto_actions=base.bounded_auto_actions,
        model_version="test-v1",
        monitor=base.monitor,
        policy=base.policy,
        governance=base.governance.__class__(
            audit_db_path=str(audit_db),
            snapshot_dir=str(snapshot_dir),
            max_snapshots=20,
            policy_version="test",
            environment="test",
        ),
        metrics=base.metrics.__class__(enabled=False, prometheus_port=9091, namespace="arl_test"),
        replay=base.replay.__class__(batch_size=16, max_steps=6, label_delay_steps=0),
        sota=base.sota,
        log_json=False,
    )
