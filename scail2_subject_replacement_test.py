"""Compatibility import and CLI for the packaged subject-replacement runner."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.pipeline.subject_replacement")
if __name__ == "__main__":
    raise SystemExit(_implementation.main())
sys.modules[__name__] = _implementation
