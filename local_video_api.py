"""Compatibility import for the packaged local-video API."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.api.local_video")
sys.modules[__name__] = _implementation
