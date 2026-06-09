from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MilestoneCheck:
    milestone_id: str
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class MilestoneStatusReport:
    checks: tuple[MilestoneCheck, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(check) for check in self.checks],
        }


def evaluate_pilot_milestones(
    *,
    dual_metric_json: dict[str, Any] | None,
    stream_records: int,
    label_delay_steps: int,
    policy_state_path: Path | None,
    min_stream_records: int = 2000,
    min_risk_reduction_pct: float = 0.0,
    target_risk_reduction_pct: float = 10.0,
    max_interventions_per_1000: float = 50.0,
) -> MilestoneStatusReport:
    checks: list[MilestoneCheck] = []

    checks.append(
        MilestoneCheck(
            "m1_stream_length",
            "M1 stream length",
            stream_records >= min_stream_records,
            f"{stream_records} records (need >= {min_stream_records})",
        )
    )
    checks.append(
        MilestoneCheck(
            "m1_label_delay",
            "M1 label delay documented",
            label_delay_steps > 0,
            f"label_delay_steps={label_delay_steps}",
        )
    )

    bounded = (dual_metric_json or {}).get("modes", {}).get("bounded_auto", {})
    summaries = {item["name"]: item for item in bounded.get("summaries", [])}
    risk_reduction = bounded.get("risk_reduction")
    utility_delta = bounded.get("utility_delta")
    buyer = bounded.get("buyer_kpis") or {}

    checks.append(
        MilestoneCheck(
            "m1_dual_modes",
            "M1 shadow + bounded_auto artifacts",
            "shadow" in (dual_metric_json or {}).get("modes", {})
            and "bounded_auto" in (dual_metric_json or {}).get("modes", {}),
            "dual_metric_report.json modes present",
        )
    )
    checks.append(
        MilestoneCheck(
            "m1_risk_reduction",
            "M1 risk reduction vs frozen",
            risk_reduction is not None and risk_reduction >= min_risk_reduction_pct / 100.0,
            f"risk_reduction={risk_reduction!r} (target >={target_risk_reduction_pct}% for external claims)",
        )
    )
    controller = summaries.get("regime_aware_delayed_bandit") or next(
        (value for key, value in summaries.items() if key != "frozen"),
        None,
    )
    interventions = (
        controller.get("bounded_interventions_per_1000", 999.0) if controller else 999.0
    )
    checks.append(
        MilestoneCheck(
            "m1_intervention_cap",
            "M1 intervention budget",
            interventions <= max_interventions_per_1000,
            f"bounded_interventions_per_1000={interventions:.1f} (cap {max_interventions_per_1000})",
        )
    )
    checks.append(
        MilestoneCheck(
            "m1_buyer_headline",
            "M1 buyer headline",
            bool(buyer.get("headline")),
            (buyer.get("headline") or "missing")[:120],
        )
    )

    utility_ok = utility_delta is not None and utility_delta >= -0.005
    checks.append(
        MilestoneCheck(
            "m2_utility",
            "M2 utility vs frozen (≥ -0.005 = equivalent)",
            utility_ok,
            f"utility_delta={utility_delta!r}",
        )
    )
    policy_ok = policy_state_path is not None and policy_state_path.exists()
    policy_detail = "policy state file missing"
    if policy_ok:
        state = json.loads(policy_state_path.read_text(encoding="utf-8"))
        policy_detail = f"kind={state.get('kind')}, matrices={len(state.get('matrices', {}))}"
    checks.append(
        MilestoneCheck(
            "m2_policy_state",
            "M2 policy persistence",
            policy_ok and policy_detail.startswith("kind="),
            policy_detail,
        )
    )

    checks.append(
        MilestoneCheck(
            "m5_buyer_kpi",
            "M5 buyer KPI block",
            bool(buyer.get("harmful_alert_reduction_pct") is not None or buyer.get("headline")),
            f"harmful_alert_reduction_pct={buyer.get('harmful_alert_reduction_pct')}",
        )
    )

    return MilestoneStatusReport(
        checks=tuple(checks),
        passed=all(check.passed for check in checks),
    )


def write_milestone_status(report: MilestoneStatusReport, output_dir: str | Path) -> Path:
    path = Path(output_dir) / "milestone_status.json"
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return path
