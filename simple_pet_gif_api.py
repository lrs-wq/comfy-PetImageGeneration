"""Compatibility import for the packaged GIF API."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.api.gif_api")
sys.modules[__name__] = _implementation
