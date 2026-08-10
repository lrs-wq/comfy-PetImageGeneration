"""Compatibility import for the packaged legacy two-action pipeline."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.legacy.two_action_pipeline")
sys.modules[__name__] = _implementation
