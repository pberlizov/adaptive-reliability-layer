#!/usr/bin/env python3
"""CMAPSS per-action ablation: which actions actually carry weight?

Runs the production benchmark (delayed_bandit) on FD001 once with all
actions enabled, then once with each action individually removed.  The
delta in terminal accuracy answers "how much does this action contribute?"

Usage
-----
    python scripts/run_cmapss_action_ablation.py
    python scripts/run_cmapss_action_ablation.py --dataset FD002 --batch-size 40
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.cmapss_benchmark import (
    CMAPSSConfig,
    run_cmapss_production_benchmark,
)
from adaptive_reliability_layer.runtime.types import DEFAULT_BOUNDED_AUTO_ACTIONS

_ALL_ACTIONS = sorted(DEFAULT_BOUNDED_AUTO_ACTIONS - {"none", "hold"})


def _run_with_actions(
    config: CMAPSSConfig,
    actions: frozenset[str],
    policy_name: str = "delayed_bandit",
) -> float:
    """Return terminal accuracy delta vs frozen for the given action set."""
    results = run_cmapss_production_benchmark(
        config,
        policy_names=["frozen", policy_name],
        bounded_auto_actions_override=actions,
    )
    frozen_final = results["frozen"].final_accuracy
    ctrl_final = results[policy_name].final_accuracy
    return ctrl_final - frozen_final


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="CMAPSS per-action ablation.")
    parser.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--policy", default="delayed_bandit", choices=["delayed_bandit", "delayed_hybrid"])
    parser.add_argument("--output-dir", default="results/cmapss_ablation")
    args = parser.parse_args(argv)

    config = CMAPSSConfig(dataset_id=args.dataset, batch_size=args.batch_size)
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    full_actions = DEFAULT_BOUNDED_AUTO_ACTIONS
    print(f"\n=== CMAPSS Action Ablation: {args.dataset} ({args.policy}) ===")
    print(f"Full action set: {sorted(full_actions)}")
    print()

    # Baseline: all actions enabled
    print("Running baseline (all actions)...", end=" ", flush=True)
    baseline_delta = _run_with_actions(config, full_actions, policy_name=args.policy)
    print(f"{baseline_delta:+.3f} pp")

    rows = [{"action_removed": "none (baseline)", "delta": baseline_delta, "cost": 0.0}]

    for action in _ALL_ACTIONS:
        ablated = frozenset(full_actions - {action})
        print(f"  Remove '{action}'...", end=" ", flush=True)
        ablated_delta = _run_with_actions(config, ablated, policy_name=args.policy)
        cost = baseline_delta - ablated_delta
        rows.append({"action_removed": action, "delta": ablated_delta, "cost": cost})
        marker = " <<< most valuable" if cost == max(r["cost"] for r in rows[1:]) and len(rows) > 1 else ""
        print(f"{ablated_delta:+.3f} pp  (cost={cost:+.3f} pp){marker}")

    # Report
    rows.sort(key=lambda r: -r["cost"])
    report_lines = [
        f"# CMAPSS Action Ablation — {args.dataset}\n",
        f"Policy: `{args.policy}` | Baseline delta: {baseline_delta:+.3f} pp\n",
        "",
        "| Action removed | Terminal Δ vs frozen | Cost vs baseline |",
        "|---|---|---|",
    ]
    for row in rows:
        sign = "+" if row["delta"] >= 0 else ""
        cost_sign = "+" if row["cost"] >= 0 else ""
        report_lines.append(
            f"| {row['action_removed']} | {sign}{row['delta']*100:.1f} pp"
            f" | {cost_sign}{row['cost']*100:.1f} pp |"
        )

    report = "\n".join(report_lines)
    print("\n" + report)

    (output_dir / f"ablation_{args.dataset}_{args.policy}.md").write_text(report + "\n", encoding="utf-8")
    (output_dir / f"ablation_{args.dataset}_{args.policy}.json").write_text(
        json.dumps({"dataset": args.dataset, "policy": args.policy, "baseline": baseline_delta, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved → {output_dir}")


if __name__ == "__main__":
    main()
