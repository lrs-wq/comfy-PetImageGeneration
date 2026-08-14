"""Compatibility bridge for incrementally modularized local-video domains."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any


def sync_local_video_runtime(
    target: dict[str, Any],
    protected: Collection[str],
) -> None:
    """Expose the facade's current dependencies to an extracted execution domain.

    The local-video API historically kept configuration, state, and helpers in one
    module.  Tests and deployment hooks patch that public facade directly.  During
    decomposition, each extracted entry point refreshes its module globals from the
    facade so those compatibility contracts remain unchanged.
    """

    from .. import local_video

    protected_values = {
        name: target[name]
        for name in protected
        if name in target
    }
    target.update(
        {
            name: value
            for name, value in vars(local_video).items()
            if not name.startswith("__")
        }
    )
    target.update(protected_values)
