"""Compatibility import and CLI for the packaged final-video audit."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.quality.final_video_fur")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
