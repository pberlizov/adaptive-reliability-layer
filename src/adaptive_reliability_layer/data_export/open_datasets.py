from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

from ..workspace import data_dir, fraud_data_dir, resolve_workspace_root
from .bundled_fraud import (
    export_ieee_from_raw,
    export_ieee_full_from_raw,
    export_ieee_synthetic,
    export_paysim_synthetic,
)
from .elliptic_baf import (
    export_baf_from_raw,
    export_baf_synthetic_fallback,
    export_elliptic_from_raw,
    export_elliptic_synthetic_fallback,
)
from .fetch_raw import fetch_baf_raw, fetch_elliptic_raw


def _openml_cache_dir(*, root: Path) -> Path:
    path = data_dir(root=root) / "openml_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def export_ulb_creditcard(*, data_id: int = 1597, root: Path | None = None) -> Path:
    workspace = root or resolve_workspace_root()
    fraud_dir = fraud_data_dir(root=workspace)
    raw_csv = fraud_dir / "creditcard_raw.csv"
    if not raw_csv.exists():
        zenodo_url = "https://zenodo.org/records/7395559/files/creditcard.csv?download=1"
        try:
            frame = pd.read_csv(zenodo_url)
            frame.to_csv(raw_csv, index=False)
        except Exception:
            cache = _openml_cache_dir(root=workspace)
            bundle = fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=str(cache))
            frame = bundle.frame if hasattr(bundle, "frame") else pd.concat([bundle.data, bundle.target], axis=1)
            frame.to_csv(raw_csv, index=False)
    else:
        frame = pd.read_csv(raw_csv)
    if "Class" in frame.columns:
        labels = frame["Class"].astype(np.int64).to_numpy()
        time_rank = frame["Time"].astype(np.int64).to_numpy() if "Time" in frame.columns else np.arange(len(labels))
        feature_cols = [column for column in frame.columns if column not in {"Class", "Time"}]
    elif "class" in frame.columns:
        labels = frame["class"].astype(np.int64).to_numpy()
        time_rank = frame["time"].astype(np.int64).to_numpy() if "time" in frame.columns else np.arange(len(labels))
        feature_cols = [column for column in frame.columns if column not in {"class", "time"}]
    else:
        raise ValueError("Unexpected credit card schema — expected Class/Time columns")
    features = frame[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    order = np.argsort(time_rank)
    out = {"time_rank": time_rank[order], "label": labels[order]}
    for index in range(scaled.shape[1]):
        out[f"feature_{index}"] = scaled[order, index]
    output = fraud_dir / "creditcard.csv"
    pd.DataFrame(out).to_csv(output, index=False)
    return output


def export_ulb_creditcard_fallback(*, rows: int = 200_000, seed: int = 17, root: Path | None = None) -> Path:
    workspace = root or resolve_workspace_root()
    rng = np.random.default_rng(seed)
    time_rank = np.sort(rng.integers(0, 1_720_000, size=rows))
    n_features = 30
    features = rng.normal(size=(rows, n_features)).astype(np.float32)
    fraud_logit = -4.2 + 0.55 * features[:, 0] - 0.35 * features[:, 1] + 2e-6 * time_rank.astype(np.float32)
    labels = (rng.random(rows) < (1.0 / (1.0 + np.exp(-fraud_logit)))).astype(np.int64)
    features[labels == 1] += rng.normal(0.7, 0.25, size=(int(labels.sum()), n_features))
    scaled = StandardScaler().fit_transform(features).astype(np.float32)
    out = {"time_rank": time_rank, "label": labels}
    for index in range(scaled.shape[1]):
        out[f"feature_{index}"] = scaled[:, index]
    output = fraud_data_dir(root=workspace) / "creditcard.csv"
    pd.DataFrame(out).to_csv(output, index=False)
    return output


def export_minimal_datasets(*, root: Path | None = None) -> Path:
    """Export PaySim synthetic only — instant, no network (Show HN quick demo)."""

    workspace = resolve_workspace_root(root)
    paysim = export_paysim_synthetic(rows=50_000, root=workspace)
    manifest: dict[str, object] = {
        "datasets": [{"id": "paysim_fraud", "path": str(paysim), "rows": 50_000, "source": "synthetic"}]
    }
    manifest_path = data_dir(root=workspace) / "open_datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path} (PaySim toy only)")
    return manifest_path


def export_open_datasets(*, root: Path | None = None) -> Path:
    workspace = resolve_workspace_root(root)
    manifest: dict[str, object] = {"datasets": []}

    paysim = export_paysim_synthetic(rows=100_000, root=workspace)
    manifest["datasets"].append({"id": "paysim_fraud", "path": str(paysim), "rows": 100_000})

    try:
        creditcard = export_ulb_creditcard(root=workspace)
        row_count = len(pd.read_csv(creditcard))
        manifest["datasets"].append(
            {"id": "ulb_creditcard_fraud", "path": str(creditcard), "rows": row_count, "source": "zenodo_or_openml"}
        )
    except Exception as exc:
        creditcard = export_ulb_creditcard_fallback(root=workspace)
        manifest["datasets"].append(
            {
                "id": "ulb_creditcard_fraud",
                "path": str(creditcard),
                "rows": 200_000,
                "source": "synthetic_fallback",
                "error": str(exc),
            }
        )
        print(f"OpenML credit card fetch failed ({exc}) — wrote synthetic fallback.")

    ieee_full = export_ieee_full_from_raw(root=workspace)
    if ieee_full is not None:
        row_count = len(pd.read_csv(ieee_full, usecols=["label"]))
        manifest["datasets"].append({"id": "ieee_cis_fraud_full", "path": str(ieee_full), "rows": row_count})
        print(f"Using IEEE-CIS full bundle ({row_count} rows).")
    else:
        ieee = export_ieee_from_raw(root=workspace)
        if ieee is None:
            ieee = export_ieee_synthetic(rows=80_000, root=workspace)
            manifest["datasets"].append({"id": "ieee_cis_fraud", "path": str(ieee), "rows": 80_000, "source": "synthetic"})
        else:
            row_count = len(pd.read_csv(ieee))
            manifest["datasets"].append({"id": "ieee_cis_fraud", "path": str(ieee), "rows": row_count, "source": "kaggle_sample"})

    elliptic = export_elliptic_from_raw(root=workspace)
    elliptic_source = "pyg_mirror"
    if elliptic is None:
        try:
            fetch_elliptic_raw(fraud_data_dir(root=workspace) / "raw" / "elliptic")
            elliptic = export_elliptic_from_raw(root=workspace)
            elliptic_source = "pyg_mirror"
        except Exception as exc:
            print(f"Elliptic public download failed ({exc}) — using synthetic fallback.")
            elliptic = export_elliptic_synthetic_fallback(root=workspace)
            elliptic_source = "synthetic_fallback"
    if elliptic is not None:
        row_count = len(pd.read_csv(elliptic, usecols=["label"]))
        manifest["datasets"].append(
            {"id": "elliptic_fraud", "path": str(elliptic), "rows": row_count, "source": elliptic_source}
        )

    baf = export_baf_from_raw(root=workspace)
    baf_source = "huggingface_mirror"
    if baf is None:
        try:
            fetch_baf_raw(fraud_data_dir(root=workspace) / "raw" / "baf")
            baf = export_baf_from_raw(root=workspace)
            baf_source = "huggingface_mirror"
        except Exception as exc:
            print(f"BAF public download failed ({exc}) — using synthetic fallback.")
            baf = export_baf_synthetic_fallback(root=workspace)
            baf_source = "synthetic_fallback"
    if baf is not None:
        row_count = len(pd.read_csv(baf, usecols=["label"]))
        manifest["datasets"].append(
            {"id": "baf_base_fraud", "path": str(baf), "rows": row_count, "source": baf_source}
        )

    manifest_path = data_dir(root=workspace) / "open_datasets_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    for item in manifest["datasets"]:
        print(f"  - {item['id']}: {item['path']} ({item.get('rows', '?')} rows)")
    return manifest_path
