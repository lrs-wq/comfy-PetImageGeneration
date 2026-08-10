"""Compatibility import for packaged uploaded-identity preparation."""

from importlib import import_module
import sys

_implementation = import_module("src.comfy_pet_pipeline.identity.uploaded_identity")
sys.modules[__name__] = _implementation
