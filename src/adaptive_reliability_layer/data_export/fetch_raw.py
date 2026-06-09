from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

ELLIPTIC_FILES = ("elliptic_txs_features.csv", "elliptic_txs_classes.csv")
PYG_BASE = "https://data.pyg.org/datasets/elliptic"
BAF_BASE_URL = (
    "https://huggingface.co/datasets/jAEhEEkIM/operationbench-baf-raw/"
    "resolve/main/Base.csv"
)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


def fetch_elliptic_raw(raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in ELLIPTIC_FILES:
        zip_path = raw_dir / f"{name}.zip"
        _download(f"{PYG_BASE}/{name}.zip", zip_path)
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)
        zip_path.unlink(missing_ok=True)
        target = raw_dir / name
        if not target.exists():
            raise FileNotFoundError(f"Expected {target} after extracting {zip_path.name}")
        written.append(target)
    return written


def fetch_baf_raw(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / "Base.csv"
    _download(BAF_BASE_URL, dest)
    return dest
