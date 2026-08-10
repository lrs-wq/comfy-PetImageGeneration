"""Compatibility import for the packaged legacy pose generator."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.legacy.pose_generator")
sys.modules[__name__] = _implementation
