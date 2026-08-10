"""Compatibility import for the packaged legacy video generator."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.legacy.video_i2v")
sys.modules[__name__] = _implementation
