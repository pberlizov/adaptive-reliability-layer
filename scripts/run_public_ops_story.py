from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.replay.ops_story import run_public_ops_story
from adaptive_reliability_layer.runtime.config import load_runtime_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ARL offline ops story on a public dataset.")
    parser.add_argument("--source-id", default="openml_electricity")
    parser.add_argument("--controller-name", default="bandit")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--stream-cycles", type=int, default=1)
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_runtime_config(args.config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else ROOT / "results" / f"ops_story_{args.source_id}"
    )
    summary = run_public_ops_story(
        source_id=args.source_id,
        runtime_config=config,
        output_dir=output_dir,
        controller_name=args.controller_name,
        steps=args.steps,
        batch_size=args.batch_size,
        stream_cycles=args.stream_cycles,
    )
    print(summary)


if __name__ == "__main__":
    main()
