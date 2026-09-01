#!/usr/bin/env python3
"""Compatibility entrypoint for the historical regression gallery."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.comfy_pet_pipeline.regression.historical_gallery import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
