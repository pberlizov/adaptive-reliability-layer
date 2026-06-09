from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def fraud_csvs():
    root = Path(__file__).resolve().parents[1]
    export = root / "scripts" / "export_bundled_fraud_data.py"
    if not (root / "data" / "fraud" / "paysim.csv").exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, str(export)], check=True, cwd=root)


def test_paysim_fraud_bundle_loads(fraud_csvs):
    from adaptive_reliability_layer.replay.real_data import load_paysim_fraud_bundle

    bundle = load_paysim_fraud_bundle(steps=4, batch_size=32, stream_cycles=1)
    assert bundle.source_id == "paysim_fraud"
    assert bundle.stream_size == 4 * 32
    assert bundle.stream.records[0].metadata.get("time_ordered") is True


def test_ieee_cis_fraud_bundle_loads(fraud_csvs):
    from adaptive_reliability_layer.replay.real_data import load_ieee_cis_fraud_bundle

    bundle = load_ieee_cis_fraud_bundle(steps=4, batch_size=32, stream_cycles=1)
    assert bundle.source_id == "ieee_cis_fraud"
    assert bundle.stream_size == 4 * 32
    root = Path(__file__).resolve().parents[1]
    full = root / "data" / "fraud" / "ieee_cis_full.csv"
    if full.exists():
        assert bundle.dataset_path is not None
        assert bundle.dataset_path.endswith("ieee_cis_full.csv")


def test_ieee_cis_fraud_torch_bundle_loads(fraud_csvs):
    from adaptive_reliability_layer.replay.real_data import load_ieee_cis_fraud_torch_bundle

    bundle = load_ieee_cis_fraud_torch_bundle(steps=4, batch_size=32, stream_cycles=1)
    assert bundle.source_id == "ieee_cis_fraud_torch"
    assert bundle.stream_size == 4 * 32
    assert bundle.stream.records[0].metadata.get("time_ordered") is True


def test_ieee_cis_fraud_torch_context_hard_bundle_loads_with_extra_features(fraud_csvs):
    from adaptive_reliability_layer.replay.real_data import (
        load_ieee_cis_fraud_torch_bundle,
        load_ieee_cis_fraud_torch_context_hard_bundle,
    )

    base = load_ieee_cis_fraud_torch_bundle(steps=4, batch_size=32, stream_cycles=1)
    context = load_ieee_cis_fraud_torch_context_hard_bundle(steps=4, batch_size=32, stream_cycles=1)
    assert context.source_id == "ieee_cis_fraud_torch_context_hard"
    assert context.feature_dim > base.feature_dim
    assert context.stream.records[0].metadata.get("time_ordered") is True


def test_elliptic_fraud_torch_context_hard_bundle_loads_with_extra_features():
    from adaptive_reliability_layer.replay.real_data import (
        load_elliptic_fraud_torch_hard_bundle,
        load_elliptic_fraud_torch_context_hard_bundle,
    )

    base = load_elliptic_fraud_torch_hard_bundle(steps=4, batch_size=32, stream_cycles=1)
    context = load_elliptic_fraud_torch_context_hard_bundle(steps=4, batch_size=32, stream_cycles=1)
    assert context.source_id == "elliptic_fraud_torch_context_hard"
    assert context.feature_dim > base.feature_dim
    assert context.stream.records[0].metadata.get("time_ordered") is True


def test_uci_gas_sensor_drift_bundle_loads():
    from adaptive_reliability_layer.replay.real_data import load_uci_gas_sensor_drift_bundle

    bundle = load_uci_gas_sensor_drift_bundle()
    assert bundle.source_id == "uci_gas_sensor_drift"
    assert bundle.stream_size > 0
    assert bundle.stream.records[0].metadata.get("time_ordered") is True
    assert bundle.stream.records[0].metadata.get("regime", "").startswith("batch")
    assert bundle.stream.records[0].metadata.get("controller_profile") == "sensor"
    assert bundle.stream.records[0].metadata.get("wedge") == "predictive_maintenance"


def test_uci_gas_sensor_drift_torch_bundle_loads():
    from adaptive_reliability_layer.replay.real_data import load_uci_gas_sensor_drift_torch_bundle

    bundle = load_uci_gas_sensor_drift_torch_bundle(epochs=2)
    assert bundle.source_id == "uci_gas_sensor_drift_torch"
    assert bundle.stream_size > 0
    assert bundle.stream.records[0].metadata.get("time_ordered") is True
    assert bundle.stream.records[0].metadata.get("regime", "").startswith("batch")
    assert bundle.stream.records[0].metadata.get("controller_profile") == "sensor"
    assert bundle.stream.records[0].metadata.get("wedge") == "predictive_maintenance"


def test_resolve_ieee_cis_csv_path_prefers_full(tmp_path: Path):
    from adaptive_reliability_layer.replay import real_data

    fraud_dir = tmp_path / "fraud"
    fraud_dir.mkdir(parents=True)
    sample = fraud_dir / "ieee_cis_sample.csv"
    full = fraud_dir / "ieee_cis_full.csv"
    sample.write_text("time_rank,label,feature_0\n1,0,0.0\n", encoding="utf-8")
    full.write_text("time_rank,label,feature_0\n1,0,0.0\n", encoding="utf-8")

    original = real_data._fraud_data_dir
    real_data._fraud_data_dir = lambda: fraud_dir
    try:
        resolved = real_data._resolve_ieee_cis_csv_path()
        assert resolved == full
    finally:
        real_data._fraud_data_dir = original


def test_resolve_ieee_cis_csv_path_falls_back_to_sample(tmp_path: Path):
    from adaptive_reliability_layer.replay import real_data

    fraud_dir = tmp_path / "fraud"
    fraud_dir.mkdir(parents=True)
    sample = fraud_dir / "ieee_cis_sample.csv"
    sample.write_text("time_rank,label,feature_0\n1,0,0.0\n", encoding="utf-8")

    original = real_data._fraud_data_dir
    real_data._fraud_data_dir = lambda: fraud_dir
    try:
        resolved = real_data._resolve_ieee_cis_csv_path()
        assert resolved == sample
    finally:
        real_data._fraud_data_dir = original
