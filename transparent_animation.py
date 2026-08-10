"""Compatibility import and CLI for packaged transparent-media helpers."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.media.transparent_animation")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
