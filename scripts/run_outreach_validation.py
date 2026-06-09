#!/usr/bin/env python3
"""Pre-outreach validation. Run from repo root or after `pip install -e .`."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from adaptive_reliability_layer.cli import outreach_validation_main

if __name__ == "__main__":
    outreach_validation_main()
