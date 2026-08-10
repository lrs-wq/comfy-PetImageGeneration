"""Compatibility import and CLI for the packaged grooming pose gate."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.quality.grooming_pose")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
