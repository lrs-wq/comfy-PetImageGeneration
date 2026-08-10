"""Compatibility import for the packaged legacy three-view generator."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.legacy.three_view_generator")
sys.modules[__name__] = _implementation
