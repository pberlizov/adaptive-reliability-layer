#!/usr/bin/env python3
"""Kafka ingest worker: consume events → ARL sidecar layer (no HTTP required)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_configs(config_path: str):
    import yaml
    from adaptive_reliability_layer.runtime.config import RuntimeConfig
    from adaptive_reliability_layer.serving.config import ServingConfig
    from adaptive_reliability_layer.ingest.kafka import KafkaIngestConfig

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    kafka_data = dict(raw.pop("kafka", None) or {})
    serving_data = dict(raw.pop("serving", None) or {})
    if serving_data.get("public_paths") is not None:
        serving_data["public_paths"] = tuple(serving_data["public_paths"])
    runtime_config = RuntimeConfig.from_mapping(raw)
    serving = ServingConfig.from_mapping(serving_data)
    kafka_config = KafkaIngestConfig.from_mapping(kafka_data)
    return runtime_config, serving, kafka_config


def _process_jsonl(path: Path, processor) -> int:
    from adaptive_reliability_layer.ingest.contract import load_events_jsonl
    from adaptive_reliability_layer.ingest.kafka import ingest_event_to_batch

    count = 0
    for event in load_events_jsonl(path):
        processor.handle_payload(
            {
                "event_id": event.event_id,
                "features": event.features.tolist(),
                "label": event.label,
                "regime_id": event.regime_id,
                "timestamp": event.timestamp,
                "metadata": event.metadata,
            },
            topic=processor.config.events_topic,
        )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Kafka ingest worker for ARL sidecar.")
    parser.add_argument("--config", default="configs/kafka_ingest_pilot.yaml")
    parser.add_argument(
        "--jsonl",
        default=None,
        help="Process JSONL file instead of Kafka (parity / offline smoke)",
    )
    parser.add_argument("--max-messages", type=int, default=None)
    parser.add_argument("--force-shadow", action="store_true")
    args = parser.parse_args()

    if args.force_shadow:
        import os

        os.environ["ARL_FORCE_SHADOW"] = "1"

    from adaptive_reliability_layer.ingest.kafka import KafkaIngestProcessor, run_kafka_consumer_loop
    from adaptive_reliability_layer.serving.loader import build_layer_for_serving
    from adaptive_reliability_layer.serving.state import ServingState

    runtime_config, serving, kafka_config = _load_configs(args.config)
    layer = build_layer_for_serving(runtime_config, serving)
    state = ServingState(layer=layer, serving=serving)
    processor = KafkaIngestProcessor(state=state, config=kafka_config)

    if args.jsonl:
        count = _process_jsonl(Path(args.jsonl), processor)
        print(f"Processed {count} events from {args.jsonl}")
        print(f"Pending delayed: {layer.pending_delayed_count}")
        return

    processed = run_kafka_consumer_loop(processor, max_messages=args.max_messages)
    print(f"Processed {processed} Kafka messages")


if __name__ == "__main__":
    main()
