#!/usr/bin/env python3
"""Replay ingested events (CSV/JSONL) through ReliabilityLayer with optional label delay."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.ingest.contract import events_to_replay_stream, load_events_csv, load_events_jsonl
from adaptive_reliability_layer.replay.dual_metric import run_dual_mode_replay, write_dual_metric_artifacts
from adaptive_reliability_layer.replay.engine import build_layer_for_tabular_replay
from adaptive_reliability_layer.runtime.config import load_runtime_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay ingest contract files through ARL.")
    parser.add_argument("--input", required=True, help="CSV or JSONL ingest file")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default="results/ingest_replay")
    parser.add_argument("--dual-mode", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.input)
    if path.suffix == ".jsonl":
        events = load_events_jsonl(path)
    else:
        events = load_events_csv(path)
    stream = events_to_replay_stream(events)

    config = load_runtime_config(args.config)
    if args.batch_size:
        config = replace(config, replay=replace(config.replay, batch_size=args.batch_size))

    if args.dual_mode:
        payload = run_dual_mode_replay(
            stream,
            runtime_config=config,
            layer_builder=build_layer_for_tabular_replay,
        )
        write_dual_metric_artifacts(payload, args.output_dir, source_label=path.name)
    else:
        from adaptive_reliability_layer.replay.engine import run_offline_replay_comparison
        from adaptive_reliability_layer.replay.report import render_replay_report

        result = run_offline_replay_comparison(stream, runtime_config=config)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "ingest_replay.md").write_text(render_replay_report(result), encoding="utf-8")

    print(f"Replay complete → {args.output_dir}")


if __name__ == "__main__":
    main()
