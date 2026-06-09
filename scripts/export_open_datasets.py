#!/usr/bin/env python3
"""Export large open datasets for production-grade ARL benchmarks."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.data_export.open_datasets import export_open_datasets


def main() -> None:
    export_open_datasets(root=ROOT)


if __name__ == "__main__":
    main()
