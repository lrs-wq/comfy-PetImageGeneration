"""Compatibility import for the packaged generation platform."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.api.platform")
sys.modules[__name__] = _implementation
