"""CLI entrypoints for commercial runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .replay.engine import build_synthetic_fraud_like_stream, export_stream_to_csv, run_offline_replay_comparison
from .replay.buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
from .replay.loader import load_replay_csv, load_replay_table, render_replay_schema_markdown
from .replay.pilot import DEFAULT_PILOT, PilotCaseStudy, run_pilot_case_study
from .replay.report import render_operator_replay_report, render_replay_report
from .runtime.config import load_runtime_config
from .workspace import data_dir, resolve_config_arg, resolve_workspace_root
from .runtime.logging_config import configure_structured_logging
from .runtime.metrics import start_metrics_server


def offline_replay_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run offline replay on historical tabular logs.")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--input", default=None, help="Canonical replay CSV or Parquet file.")
    parser.add_argument("--csv", default=None, help="Backward-compatible alias for --input.")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--strategies", default="frozen,naive,controller,bandit")
    parser.add_argument("--output-dir", default="results/offline_replay")
    args = parser.parse_args(argv)

    config = load_runtime_config(resolve_config_arg(args.config))
    configure_structured_logging(json_logs=config.log_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        stream = build_synthetic_fraud_like_stream(
            steps=config.replay.max_steps or 90,
            batch_size=config.replay.batch_size,
        )
        export_stream_to_csv(stream, output_dir / "stream.csv")
        input_path = str(output_dir / "stream.csv")
    elif args.input or args.csv:
        input_path = args.input or args.csv
        stream = load_replay_table(input_path, config.replay)
    else:
        raise SystemExit("Provide --input/--csv or --synthetic")

    strategies = tuple(part.strip() for part in args.strategies.split(",") if part.strip())
    result = run_offline_replay_comparison(stream, runtime_config=config, strategies=strategies)
    report = render_replay_report(result)
    operator_report = render_operator_replay_report(result)
    buyer_report = render_buyer_replay_report(result, source_label=str(input_path), wedge="generic_mlops")
    buyer_kpis = compute_buyer_kpis(result)
    (output_dir / "offline_replay.md").write_text(report, encoding="utf-8")
    (output_dir / "offline_replay_operator.md").write_text(operator_report, encoding="utf-8")
    (output_dir / "offline_replay_buyer.md").write_text(buyer_report, encoding="utf-8")
    (output_dir / "replay_schema.md").write_text(
        render_replay_schema_markdown(config.replay.feature_prefix),
        encoding="utf-8",
    )
    (output_dir / "offline_replay.json").write_text(
        json.dumps(
            {
                "summaries": [summary.__dict__ for summary in result.summaries],
                "controller_vs_frozen_utility_delta": result.controller_vs_frozen_utility_delta,
                "controller_vs_frozen_risk_reduction": result.controller_vs_frozen_risk_reduction,
                "controller_vs_frozen_harmful_events_avoided": result.controller_vs_frozen_harmful_events_avoided,
                "controller_vs_frozen_retrain_deferral_steps": result.controller_vs_frozen_retrain_deferral_steps,
                "buyer_kpis": buyer_kpis.__dict__ if buyer_kpis is not None else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)


def pilot_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a commercial pilot case study offline replay.")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--output-dir", default="results/pilot")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--label-delay-steps", type=int, default=None)
    parser.add_argument("--operating-mode", default=None, choices=["shadow", "recommend", "bounded_auto"])
    args = parser.parse_args(argv)

    config = load_runtime_config(resolve_config_arg(args.config))
    configure_structured_logging(json_logs=config.log_json)
    pilot = DEFAULT_PILOT
    if args.csv or args.label_delay_steps is not None or args.operating_mode is not None:
        pilot = PilotCaseStudy(
            name=pilot.name,
            wedge=pilot.wedge,
            description=pilot.description,
            primary_kpi=pilot.primary_kpi,
            dataset_path=args.csv,
            use_synthetic_stream=args.csv is None,
            label_delay_steps=pilot.label_delay_steps if args.label_delay_steps is None else args.label_delay_steps,
            operating_mode=args.operating_mode or pilot.operating_mode,
            strategies=pilot.strategies,
            dual_mode=pilot.dual_mode,
            controller_name=pilot.controller_name,
        )
    layer_builder = None
    config_path = str(args.config).lower()
    if "paysim" in config_path or "pilot_fraud_torch" in config_path:
        from .replay.real_data import load_paysim_fraud_torch_bundle

        bundle = load_paysim_fraud_torch_bundle(
            steps=config.replay.max_steps or 48,
            batch_size=config.replay.batch_size,
        )
        layer_builder = bundle.build_layer
    summary = run_pilot_case_study(
        pilot,
        runtime_config=config,
        output_dir=args.output_dir,
        layer_builder=layer_builder,
    )
    print("Pilot case study complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def metrics_main(argv: list[str] | None = None) -> None:
    import time

    parser = argparse.ArgumentParser(description="Start Prometheus metrics endpoint for ARL runtime.")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_runtime_config(resolve_config_arg(args.config))
    port = args.port or config.metrics.prometheus_port
    start_metrics_server(port)
    print(f"Prometheus metrics listening on :{port}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down metrics server.")


def real_data_verification_main(argv: list[str] | None = None) -> None:
    import yaml

    from .replay.verification_suite import render_verification_suite_report, run_real_data_verification_suite

    parser = argparse.ArgumentParser(description="Verify ARL runtime on multiple real public datasets.")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--output-dir", default="results/real_data_verification")
    parser.add_argument("--sources", default=None, help="Comma-separated source ids")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args(argv)

    config = load_runtime_config(resolve_config_arg(args.config))
    configure_structured_logging(json_logs=config.log_json)

    with resolve_config_arg(args.config).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    source_ids = None
    if args.sources:
        source_ids = tuple(part.strip() for part in args.sources.split(",") if part.strip())
    elif raw.get("sources"):
        source_ids = tuple(raw["sources"])

    suite = run_real_data_verification_suite(
        runtime_config=config,
        source_ids=source_ids,
        output_dir=args.output_dir,
        skip_on_error=not args.fail_fast,
    )
    print(render_verification_suite_report(suite, list(suite.errors)))
    if not suite.passed:
        raise SystemExit(1)


def outreach_validation_main(argv: list[str] | None = None) -> None:
    import json
    from dataclasses import asdict, replace

    from .replay.buyer_kpis import compute_buyer_kpis, render_buyer_replay_report
    from .replay.engine import run_offline_replay_comparison
    from .replay.real_data import load_real_data_bundle
    from .replay.tta_comparison import render_tta_comparison_report, run_tta_tabular_comparison

    parser = argparse.ArgumentParser(description="Run outreach validation artifacts.")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--output-dir", default="results/outreach_validation")
    parser.add_argument("--credit-cycles", type=int, default=6)
    parser.add_argument("--tta-steps", type=int, default=90)
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = load_runtime_config(resolve_config_arg(args.config))
    configure_structured_logging(json_logs=config.log_json)

    credit_steps = config.replay.max_steps or 18
    effective_steps = credit_steps * args.credit_cycles
    credit_config = replace(
        config,
        replay=replace(config.replay, max_steps=effective_steps),
    )
    credit_bundle = load_real_data_bundle(
        "openml_credit_g",
        steps=credit_steps,
        batch_size=config.replay.batch_size,
        stream_cycles=args.credit_cycles,
    )
    credit_replay = run_offline_replay_comparison(
        credit_bundle.stream,
        runtime_config=credit_config,
        strategies=("frozen", "naive", "bandit"),
        layer_builder=credit_bundle.build_layer,
    )
    credit_buyer = render_buyer_replay_report(
        credit_replay,
        source_label=(
            f"German Credit (OpenML), {credit_bundle.stream_size} streamed rows, "
            f"{args.credit_cycles} regime cycles"
        ),
        wedge="fraud_risk",
    )
    (output / "german_credit_buyer_report.md").write_text(credit_buyer, encoding="utf-8")
    credit_kpis = compute_buyer_kpis(credit_replay, controller_name="bandit")
    (output / "german_credit_summary.json").write_text(
        json.dumps(
            {
                "source_id": "openml_credit_g",
                "stream_rows": credit_bundle.stream_size,
                "stream_cycles": args.credit_cycles,
                "buyer_kpis": asdict(credit_kpis) if credit_kpis else None,
                "replay": {
                    "utility_delta": credit_replay.controller_vs_frozen_utility_delta,
                    "risk_reduction": credit_replay.controller_vs_frozen_risk_reduction,
                    "summaries": [summary.__dict__ for summary in credit_replay.summaries],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    tta_result = run_tta_tabular_comparison(steps=args.tta_steps, batch_size=48, seed=7)
    (output / "tta_baseline_comparison.md").write_text(
        render_tta_comparison_report(tta_result),
        encoding="utf-8",
    )

    print(f"Wrote outreach artifacts to {output.resolve()}")
    print("--- German Credit buyer headline ---")
    if credit_kpis:
        print(credit_kpis.headline)
    print("--- TTA comparison saved ---")
    print(output / "tta_baseline_comparison.md")


def serve_main(argv: list[str] | None = None) -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Serve Adaptive Reliability Layer over HTTP.")
    parser.add_argument("--config", default="serving_pilot_fraud_torch.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--model-bundle", default=None)
    parser.add_argument("--force-shadow", action="store_true")
    args = parser.parse_args(argv)

    if args.force_shadow:
        os.environ["ARL_FORCE_SHADOW"] = "1"

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Install serving extras: pip install -e '.[serving,prometheus]'") from exc

    from .runtime.config import RuntimeConfig
    from .serving.app import create_app
    from .serving.config import ServingConfig, load_serving_config_from_yaml
    from .serving.loader import build_layer_for_serving

    config_path = resolve_config_arg(args.config)
    raw, serving = load_serving_config_from_yaml(config_path)
    if args.model_bundle:
        serving = ServingConfig(**{**serving.__dict__, "model_bundle": args.model_bundle})
    runtime_config = RuntimeConfig.from_mapping(raw)
    layer = build_layer_for_serving(runtime_config, serving)
    app = create_app(config_path=str(config_path), layer=layer, serving=serving)
    uvicorn.run(app, host=args.host, port=args.port)


def customer_replay_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run a design-partner customer CSV/JSONL replay (shadow-first deliverables)."
    )
    parser.add_argument("--input", required=True, help="Customer CSV or JSONL (ingest contract)")
    parser.add_argument("--config", default="customer_shadow.yaml")
    parser.add_argument("--output-dir", default="results/customer_replay")
    parser.add_argument("--customer", default="customer", help="Label for reports and manifest")
    parser.add_argument("--wedge", default="fraud_risk", choices=("fraud_risk", "generic_mlops", "sensor_ops"))
    parser.add_argument("--dual-mode", action="store_true", default=True)
    parser.add_argument("--single-mode", action="store_true", help="Disable shadow vs bounded_auto dual report")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--label-delay-steps", type=int, default=None)
    args = parser.parse_args(argv)

    from .replay.customer_replay import CustomerReplaySpec, run_customer_replay

    result = run_customer_replay(
        CustomerReplaySpec(
            input_path=Path(args.input),
            config_path=resolve_config_arg(args.config),
            output_dir=Path(args.output_dir),
            customer_label=args.customer,
            wedge=args.wedge,
            dual_mode=not args.single_mode,
            batch_size=args.batch_size,
            label_delay_steps=args.label_delay_steps,
        )
    )
    print(
        f"Customer replay complete → {result.output_dir}\n"
        f"utility_delta={result.controller_vs_frozen_utility_delta!r} "
        f"risk_reduction={result.controller_vs_frozen_risk_reduction!r}"
    )


def demo_main(argv: list[str] | None = None) -> None:
    """Show HN toy demo — alias for `arl-hn-launch --quick`."""

    hn_launch_main(["--quick", *(argv or [])])


def export_datasets_main(argv: list[str] | None = None) -> None:
    from .data_export.open_datasets import export_open_datasets

    argparse.ArgumentParser(description="Export public fraud CSVs into ./data/fraud").parse_args(argv)
    export_open_datasets()


def hn_launch_main(argv: list[str] | None = None) -> None:
    import sys

    from .replay.hn_launch import (
        export_public_datasets,
        render_hn_comparison_table,
        run_hn_discrimination_benchmark,
        run_hn_production_benchmark,
        verify_sidecar_health,
        write_hn_launch_artifacts,
    )

    root = resolve_workspace_root()
    parser = argparse.ArgumentParser(description="HN launch: export data, benchmarks, comparison table.")
    parser.add_argument("--output-dir", default="results/hn_launch")
    parser.add_argument("--production-config", default="hn_launch_production.yaml")
    parser.add_argument("--discrimination-config", default="hn_launch_discrimination.yaml")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-production", action="store_true")
    parser.add_argument("--skip-discrimination", action="store_true")
    parser.add_argument("--skip-sidecar", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Toy demo: PaySim synthetic only, ~2–5 min (no dataset downloads)",
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.production_config = "hn_launch_quick.yaml"
        args.discrimination_config = "hn_launch_discrimination_quick.yaml"

    output_dir = root / args.output_dir
    manifest_path = None
    if not args.skip_export:
        print("==> Exporting public fraud datasets..." + (" (PaySim toy only)" if args.quick else ""))
        manifest_path = export_public_datasets(root=root, minimal=args.quick)
    else:
        candidate = data_dir(root=root) / "open_datasets_manifest.json"
        if candidate.exists():
            manifest_path = candidate
    if args.export_only:
        print("Export-only complete.")
        return

    production_report = None
    discrimination_report = None
    if not args.skip_production:
        print("==> Production claim benchmark...")
        production_report = run_hn_production_benchmark(
            config_path=resolve_config_arg(args.production_config, root=root),
            output_dir=output_dir / "production",
        )
    if not args.skip_discrimination:
        print("==> Hard-slice discrimination benchmark...")
        discrimination_report = run_hn_discrimination_benchmark(
            config_path=resolve_config_arg(args.discrimination_config, root=root),
            output_dir=output_dir / "discrimination",
        )
    sidecar_ok = None
    if not args.skip_sidecar:
        sidecar_ok = verify_sidecar_health()
        print(f"Sidecar OK: {sidecar_ok}")

    paths = write_hn_launch_artifacts(
        output_dir=output_dir,
        production=production_report,
        discrimination=discrimination_report,
        manifest_path=manifest_path,
        sidecar_ok=sidecar_ok,
        quick=args.quick,
    )
    print(render_hn_comparison_table(production=production_report, discrimination=discrimination_report, quick=args.quick))
    print(f"\nWrote {paths['comparison_table']}")


def fraud_public_benchmark_main(argv: list[str] | None = None) -> None:
    import argparse

    from .replay.fraud_public_benchmark import run_fraud_public_benchmark

    parser = argparse.ArgumentParser(description="Run public fraud benchmark suite.")
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--output-dir", default="results/fraud_public_benchmark")
    parser.add_argument("--stream-cycles", type=int, default=6)
    parser.add_argument("--skip-torch-full", action="store_true")
    args = parser.parse_args(argv)
    run_fraud_public_benchmark(
        config_path=args.config,
        output_dir=args.output_dir,
        stream_cycles=args.stream_cycles,
        skip_torch_full=args.skip_torch_full,
    )
