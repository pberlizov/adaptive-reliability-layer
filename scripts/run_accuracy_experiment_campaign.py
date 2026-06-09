#!/usr/bin/env python3
"""Bounded accuracy experiment campaign on hard fraud temporal slices."""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from adaptive_reliability_layer.replay.discrimination_benchmark import (
    run_discrimination_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "accuracy_experiment" / "base_hard_fraud.yaml"
DEFAULT_OUTPUT = ROOT / "results" / "accuracy_experiment_campaign"

HARD_FRAUD_SOURCES = (
    "ieee_cis_fraud_torch_hard",
    "ulb_creditcard_fraud_torch_hard",
    "paysim_fraud_torch_hard",
)

PRIMARY_SOURCES_FOR_STOP = ("ieee_cis_fraud_torch_hard", "ulb_creditcard_fraud_torch_hard")
MIN_BAL_ACC_GAIN = 0.01
MIN_EXPERIMENTS_BEFORE_EARLY_STOP = 4
MAX_EXPERIMENTS = 6
MAX_WALL_CLOCK_SECONDS = 2 * 60 * 60


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    hypothesis: str
    candidate_strategy: str
    config_patch: dict[str, Any]


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        id="exp01_correction_plus_governor",
        hypothesis=(
            "Delayed residual correction + governor (no explicit model actions) improves "
            "minority recall on hard temporal slices when label delay reveals prevalence shift."
        ),
        candidate_strategy="correction_plus_governor",
        config_patch={},
    ),
    ExperimentSpec(
        id="exp02_sensitive_monitor_cpg",
        hypothesis=(
            "Lower shift-monitor thresholds fire correction earlier on IEEE/PaySim hard tails, "
            "improving balanced accuracy before late-stream recall collapse."
        ),
        candidate_strategy="correction_plus_governor",
        config_patch={
            "monitor": {"alert_threshold": 0.90, "severe_threshold": 1.35},
        },
    ),
    ExperimentSpec(
        id="exp03_kpi_recall_oriented_cpg",
        hypothesis=(
            "Recall-oriented KPI (lower false-alert cost) steers delayed threshold learning "
            "toward catching more fraud at acceptable precision cost."
        ),
        candidate_strategy="correction_plus_governor",
        config_patch={
            "kpi": {"false_alert_cost": 0.02, "retrain_recommendation_cost": 0.12},
        },
    ),
    ExperimentSpec(
        id="exp04_strong_threshold_lr_cpg",
        hypothesis=(
            "Stronger posterior-driven threshold learning (3× default rate) lowers decision "
            "threshold when delayed labels reveal under-predicted fraud rate."
        ),
        candidate_strategy="correction_plus_governor",
        config_patch={
            "policy": {"threshold_learning_rate": 0.30},
        },
    ),
    ExperimentSpec(
        id="exp05_label_shift_bounded_cpg",
        hypothesis=(
            "Allowing label_shift (prevalence correction) alongside governor-only correction "
            "helps hard slices where positive rate drifts across temporal halves."
        ),
        candidate_strategy="correction_plus_governor",
        config_patch={
            "bounded_auto_actions": ["none", "hold", "label_shift"],
        },
    ),
    ExperimentSpec(
        id="exp06_proactive_drift_cpg",
        hypothesis=(
            "Proactive drift hold + conformal coverage + correction path lowers threshold "
            "under confirmed shift on IEEE/PaySim without unsafe adaptation."
        ),
        candidate_strategy="correction_plus_governor",
        config_patch={
            "sota": {"proactive_drift_enabled": True},
        },
    ),
)


def _deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _write_experiment_config(
    spec: ExperimentSpec,
    *,
    config_dir: Path,
) -> Path:
    base = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    merged = _deep_merge(base, spec.config_patch)
    benchmark = merged.setdefault("discrimination_benchmark", {})
    strategies = list(benchmark.get("strategies", ["frozen"]))
    if spec.candidate_strategy not in strategies:
        strategies.append(spec.candidate_strategy)
    benchmark["strategies"] = strategies
    path = config_dir / f"{spec.id}.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return path


def _strategy_metrics(report: Any, source_id: str, strategy: str) -> dict[str, Any]:
    for source in report.sources:
        if source.source_id != source_id:
            continue
        for item in source.strategy_metrics:
            if item.name == strategy:
                metrics = item.stream_metrics
                return {
                    "balanced_accuracy": metrics.balanced_accuracy,
                    "pr_auc": metrics.pr_auc,
                    "recall": metrics.recall,
                    "recall_at_precision_80": metrics.recall_at_precision_80,
                    "cost_weighted_error": metrics.cost_weighted_error,
                    "accuracy": metrics.accuracy,
                    "mean_decision_threshold": item.mean_decision_threshold,
                    "mean_correction_applied_rate": item.mean_correction_applied_rate,
                }
    raise KeyError(f"strategy {strategy!r} not found for source {source_id!r}")


def _delta_vs_frozen(
    report: Any,
    *,
    candidate: str,
    source_id: str,
) -> dict[str, float | None]:
    frozen = _strategy_metrics(report, source_id, "frozen")
    cand = _strategy_metrics(report, source_id, candidate)
    out: dict[str, float | None] = {}
    for key in (
        "balanced_accuracy",
        "pr_auc",
        "recall",
        "recall_at_precision_80",
        "cost_weighted_error",
        "accuracy",
    ):
        f_val = frozen.get(key)
        c_val = cand.get(key)
        if f_val is None or c_val is None:
            out[f"delta_{key}"] = None
        else:
            out[f"delta_{key}"] = float(c_val) - float(f_val)
    out["delta_mean_decision_threshold"] = float(cand["mean_decision_threshold"]) - float(
        frozen["mean_decision_threshold"]
    )
    return out


def _meets_stop_rule(experiments: list[dict[str, Any]]) -> tuple[bool, str]:
    if len(experiments) >= MAX_EXPERIMENTS:
        return True, f"completed {MAX_EXPERIMENTS} experiment configs"
    best_ieee = max(
        (exp["deltas"]["ieee_cis_fraud_torch_hard"]["delta_balanced_accuracy"] or -999.0 for exp in experiments),
        default=-999.0,
    )
    best_ulb = max(
        (
            exp["deltas"]["ulb_creditcard_fraud_torch_hard"]["delta_balanced_accuracy"] or -999.0
            for exp in experiments
        ),
        default=-999.0,
    )
    if len(experiments) >= MIN_EXPERIMENTS_BEFORE_EARLY_STOP and best_ieee < MIN_BAL_ACC_GAIN and best_ulb < MIN_BAL_ACC_GAIN:
        return True, (
            f"no improvement: best bal_acc gain vs frozen < {MIN_BAL_ACC_GAIN:.2f} "
            f"on both ieee_hard ({best_ieee:+.4f}) and ulb_hard ({best_ulb:+.4f})"
        )
    return False, ""


def _render_report(
    *,
    experiments: list[dict[str, Any]],
    baseline: dict[str, Any],
    stop_reason: str,
    wall_clock_seconds: float,
    started_at: str,
) -> str:
    lines = [
        "# Accuracy experiment campaign",
        "",
        f"**Started:** {started_at}",
        f"**Wall clock:** {wall_clock_seconds / 60:.1f} min",
        f"**Stop reason:** {stop_reason}",
        "",
        "## Charter",
        "",
        "Primary metrics (ranked): balanced_accuracy, PR-AUC, recall@precision≥0.80, cost_weighted_error.",
        "Hard temporal slices only: IEEE, ULB, PaySim `*_torch_hard`.",
        "",
        "## Frozen baseline (reference)",
        "",
        "| source | bal_acc | PR-AUC | recall | R@P≥0.8 | cost err | threshold |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source_id in HARD_FRAUD_SOURCES:
        row = baseline[source_id]
        r_at_p = row["recall_at_precision_80"]
        r_at_p_text = "n/a" if r_at_p is None else f"{r_at_p:.3f}"
        pr_auc = row["pr_auc"]
        pr_auc_text = "n/a" if pr_auc is None else f"{pr_auc:.3f}"
        lines.append(
            f"| `{source_id}` | {row['balanced_accuracy']:.3f} | {pr_auc_text} | "
            f"{row['recall']:.3f} | {r_at_p_text} | {row['cost_weighted_error']:.3f} | "
            f"{row['mean_decision_threshold']:.3f} |"
        )
    lines.extend(["", "## Experiments", ""])
    for exp in experiments:
        lines.extend(
            [
                f"### {exp['id']}",
                "",
                f"**Hypothesis:** {exp['hypothesis']}",
                f"**Candidate:** `{exp['candidate_strategy']}`",
                f"**Config:** `{exp['config_path']}`",
                "",
                "| source | Δ bal_acc | Δ PR-AUC | Δ recall | Δ R@P≥0.8 | Δ cost err | Δ threshold |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for source_id in HARD_FRAUD_SOURCES:
            deltas = exp["deltas"][source_id]
            lines.append(
                "| `{source}` | {dbal} | {dpr} | {drec} | {drp} | {dcost} | {dthr} |".format(
                    source=source_id,
                    dbal=_fmt_delta(deltas.get("delta_balanced_accuracy")),
                    dpr=_fmt_delta(deltas.get("delta_pr_auc")),
                    drec=_fmt_delta(deltas.get("delta_recall")),
                    drp=_fmt_delta(deltas.get("delta_recall_at_precision_80")),
                    dcost=_fmt_delta(deltas.get("delta_cost_weighted_error")),
                    dthr=_fmt_delta(deltas.get("delta_mean_decision_threshold")),
                )
            )
        lines.append("")
    best = _best_experiment(experiments)
    lines.extend(
        [
            "## Best deltas vs frozen",
            "",
            f"- **Best experiment:** `{best['id']}`",
            f"- **IEEE hard Δ bal_acc:** {_fmt_delta(best['deltas']['ieee_cis_fraud_torch_hard'].get('delta_balanced_accuracy'))}",
            f"- **ULB hard Δ bal_acc:** {_fmt_delta(best['deltas']['ulb_creditcard_fraud_torch_hard'].get('delta_balanced_accuracy'))}",
            f"- **PaySim hard Δ bal_acc:** {_fmt_delta(best['deltas']['paysim_fraud_torch_hard'].get('delta_balanced_accuracy'))}",
            "",
            "## Recommendation",
            "",
            _recommendation(experiments, best),
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_delta(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def _best_experiment(experiments: list[dict[str, Any]]) -> dict[str, Any]:
    def score(exp: dict[str, Any]) -> float:
        ieee = exp["deltas"]["ieee_cis_fraud_torch_hard"].get("delta_balanced_accuracy") or -999.0
        ulb = exp["deltas"]["ulb_creditcard_fraud_torch_hard"].get("delta_balanced_accuracy") or -999.0
        paysim = exp["deltas"]["paysim_fraud_torch_hard"].get("delta_balanced_accuracy") or -999.0
        pr_ieee = exp["deltas"]["ieee_cis_fraud_torch_hard"].get("delta_pr_auc") or 0.0
        return ieee + ulb + paysim + 0.5 * pr_ieee

    return max(experiments, key=score)


def _recommendation(experiments: list[dict[str, Any]], best: dict[str, Any]) -> str:
    ieee_gain = best["deltas"]["ieee_cis_fraud_torch_hard"].get("delta_balanced_accuracy") or 0.0
    ulb_gain = best["deltas"]["ulb_creditcard_fraud_torch_hard"].get("delta_balanced_accuracy") or 0.0
    if ieee_gain >= MIN_BAL_ACC_GAIN or ulb_gain >= MIN_BAL_ACC_GAIN:
        return (
            f"Continue accuracy track in main thread for `{best['id']}` — at least one primary hard slice "
            "shows ≥1pp balanced-accuracy gain. Validate on customer replay before production claims."
        )
    return (
        "**Do not invest in flagship accuracy; invest in utility vs scheduled retrain, "
        "correction+governor operational story, and revealed-loss / chargeback-weighted KPIs instead.** "
        "Hard-slice mechanism tweaks (monitor sensitivity, KPI reweight, threshold LR, label_shift, "
        "proactive drift) did not move balanced accuracy meaningfully on IEEE or ULB."
    )


def run_campaign(
    *,
    output_dir: Path = DEFAULT_OUTPUT,
    max_experiments: int = MAX_EXPERIMENTS,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir = output_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    baseline_config = _write_experiment_config(
        ExperimentSpec(
            id="baseline_frozen",
            hypothesis="Frozen reference on hard fraud trio.",
            candidate_strategy="frozen",
            config_patch={},
        ),
        config_dir=config_dir,
    )
    baseline_report = run_discrimination_benchmark(config_path=baseline_config)
    baseline = {
        source_id: _strategy_metrics(baseline_report, source_id, "frozen")
        for source_id in HARD_FRAUD_SOURCES
    }

    experiments: list[dict[str, Any]] = []
    stop_reason = f"completed {max_experiments} experiment configs"
    for spec in EXPERIMENTS[:max_experiments]:
        elapsed = time.monotonic() - started
        if elapsed >= MAX_WALL_CLOCK_SECONDS:
            stop_reason = f"2-hour wall clock limit ({elapsed / 60:.1f} min elapsed)"
            break
        config_path = _write_experiment_config(spec, config_dir=config_dir)
        report = run_discrimination_benchmark(config_path=config_path)
        deltas = {
            source_id: _delta_vs_frozen(report, candidate=spec.candidate_strategy, source_id=source_id)
            for source_id in HARD_FRAUD_SOURCES
        }
        experiments.append(
            {
                "id": spec.id,
                "hypothesis": spec.hypothesis,
                "candidate_strategy": spec.candidate_strategy,
                "config_path": str(config_path.resolve().relative_to(ROOT.resolve())),
                "deltas": deltas,
                "candidate_metrics": {
                    source_id: _strategy_metrics(report, source_id, spec.candidate_strategy)
                    for source_id in HARD_FRAUD_SOURCES
                },
            }
        )
        should_stop, early_reason = _meets_stop_rule(experiments)
        if should_stop and early_reason.startswith("no improvement"):
            stop_reason = early_reason
            break

    wall_clock_seconds = time.monotonic() - started
    if wall_clock_seconds >= MAX_WALL_CLOCK_SECONDS and "wall clock" not in stop_reason:
        stop_reason = f"2-hour wall clock limit ({wall_clock_seconds / 60:.1f} min elapsed)"

    summary = {
        "started_at": started_at,
        "wall_clock_seconds": wall_clock_seconds,
        "stop_reason": stop_reason,
        "experiments_run": len(experiments),
        "baseline": baseline,
        "experiments": experiments,
        "best_experiment_id": _best_experiment(experiments)["id"] if experiments else None,
        "recommendation": _recommendation(experiments, _best_experiment(experiments))
        if experiments
        else "No experiments completed.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(
        _render_report(
            experiments=experiments,
            baseline=baseline,
            stop_reason=stop_reason,
            wall_clock_seconds=wall_clock_seconds,
            started_at=started_at,
        ),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory")
    parser.add_argument("--max-experiments", type=int, default=MAX_EXPERIMENTS)
    args = parser.parse_args()
    summary = run_campaign(output_dir=Path(args.output), max_experiments=args.max_experiments)
    print(json.dumps({k: summary[k] for k in ("stop_reason", "experiments_run", "best_experiment_id")}, indent=2))
    print(f"\nWrote {Path(args.output) / 'report.md'}")


if __name__ == "__main__":
    main()
