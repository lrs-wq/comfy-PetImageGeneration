"""Compatibility import and CLI for the packaged fur-surface gate."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.quality.fur_surface")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
