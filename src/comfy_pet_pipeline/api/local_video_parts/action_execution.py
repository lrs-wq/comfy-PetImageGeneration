"""Species/action execution boundaries for the local-video pipeline.

The GPU worker is intentionally single threaded, but each requested action must
still behave like an independent unit of work.  This module provides the small,
pure boundary used by both the Qwen reference stage and the SCAIL animation
stage: stable action identities, isolated configuration copies, and an
exception barrier that lets sibling actions continue before the batch reports
its failures.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from types import TracebackType
from typing import Any


@dataclass(frozen=True)
class ActionExecutionScope:
    """Immutable identity for one species/action execution."""

    species: str
    action: str
    request_index: int
    execution_index: int

    @property
    def isolation_key(self) -> str:
        return f"{self.species}:{self.action}"

    def public(self) -> dict[str, Any]:
        return {
            "isolation_key": self.isolation_key,
            "species": self.species,
            "action": self.action,
            "request_index": self.request_index,
            "execution_index": self.execution_index,
        }


@dataclass(frozen=True)
class ActionStageFailure:
    """Serializable failure belonging to exactly one action and stage."""

    scope: ActionExecutionScope
    stage: str
    error_type: str
    error: str

    @classmethod
    def from_exception(
        cls,
        scope: ActionExecutionScope,
        stage: str,
        error: Exception,
    ) -> ActionStageFailure:
        return cls(
            scope=scope,
            stage=stage,
            error_type=type(error).__name__,
            error=str(error)[:4000],
        )

    def public(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scope"]["isolation_key"] = self.scope.isolation_key
        return payload


class IsolatedActionStageError(RuntimeError):
    """Raised after all runnable sibling actions have had their turn."""

    def __init__(self, stage: str, failures: Sequence[ActionStageFailure]) -> None:
        self.stage = stage
        self.failures = tuple(failures)
        summary = "; ".join(
            f"{failure.scope.isolation_key}={failure.error_type}: {failure.error}"
            for failure in self.failures
        )
        super().__init__(f"{stage} failed for isolated action(s): {summary}")


def build_action_execution_scopes(
    *,
    species: str,
    requested_actions: Sequence[str],
    execution_actions: Sequence[str] | None = None,
) -> tuple[ActionExecutionScope, ...]:
    """Build a validated action plan without relying on loop position.

    ``execution_actions`` may be a subset (for example, actions whose Qwen
    reference already degraded before SCAIL), but it cannot add or duplicate an
    action.  Request indices remain stable even when execution order differs.
    """

    normalized_species = str(species).strip().lower()
    if normalized_species not in {"cat", "dog"}:
        raise ValueError(f"Unsupported subject species: {species!r}")
    requested = tuple(str(action).strip().lower() for action in requested_actions)
    if not requested or any(not action for action in requested):
        raise ValueError("At least one non-empty action is required.")
    if len(set(requested)) != len(requested):
        raise ValueError("Requested actions must be unique within one job.")
    scheduled = tuple(
        str(action).strip().lower()
        for action in (
            requested if execution_actions is None else execution_actions
        )
    )
    if len(set(scheduled)) != len(scheduled):
        raise ValueError("Execution actions must be unique within one stage.")
    unknown = [action for action in scheduled if action not in requested]
    if unknown:
        raise ValueError(
            "Execution actions must be a subset of requested actions: "
            + ", ".join(unknown)
        )
    request_indices = {
        action: index for index, action in enumerate(requested, start=1)
    }
    return tuple(
        ActionExecutionScope(
            species=normalized_species,
            action=action,
            request_index=request_indices[action],
            execution_index=execution_index,
        )
        for execution_index, action in enumerate(scheduled, start=1)
    )


def isolated_action_config(
    effective_config: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    """Return a deep action snapshot so handlers cannot mutate siblings."""

    raw_actions = effective_config.get("actions")
    if not isinstance(raw_actions, Mapping):
        raise RuntimeError("Effective subject config lacks per-action settings.")
    raw_action = raw_actions.get(action)
    if not isinstance(raw_action, Mapping):
        raise RuntimeError(f"Effective subject config lacks action {action!r}.")
    return deepcopy(dict(raw_action))


def isolated_effective_action_config(
    effective_config: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    """Snapshot shared settings while exposing only the selected action.

    Action handlers still need job-wide frame/fps/model settings.  Giving them
    a private copy with a one-entry ``actions`` map prevents an accidental
    write or lookup from crossing into a sibling action.
    """

    snapshot = deepcopy(dict(effective_config))
    snapshot["actions"] = {
        action: isolated_action_config(effective_config, action)
    }
    return snapshot


class isolate_action_stage:
    """Suppress one ordinary action failure while preserving fatal exceptions.

    The caller must invoke :func:`raise_for_action_stage_failures` after all
    scopes have run.  This means a failed action cannot prevent its siblings
    from producing outputs, while the enclosing job still fails closed if the
    action could not be recovered.
    """

    def __init__(
        self,
        *,
        scope: ActionExecutionScope,
        stage: str,
        failures: list[ActionStageFailure],
        fatal_exceptions: tuple[type[BaseException], ...] = (),
        on_failure: Callable[[ActionStageFailure, Exception], None] | None = None,
        enabled: bool = True,
    ) -> None:
        self.scope = scope
        self.stage = stage
        self.failures = failures
        self.fatal_exceptions = fatal_exceptions
        self.on_failure = on_failure
        self.enabled = enabled

    def __enter__(self) -> ActionExecutionScope:
        return self.scope

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del error_type, traceback
        if error is None or not self.enabled:
            return False
        if self.fatal_exceptions and isinstance(error, self.fatal_exceptions):
            return False
        if not isinstance(error, Exception):
            return False
        failure = ActionStageFailure.from_exception(
            self.scope,
            self.stage,
            error,
        )
        self.failures.append(failure)
        if self.on_failure is not None:
            self.on_failure(failure, error)
        return True


def failures_by_action(
    failures: Iterable[ActionStageFailure],
) -> dict[str, dict[str, Any]]:
    return {
        failure.scope.action: failure.public()
        for failure in failures
    }


def raise_for_action_stage_failures(
    stage: str,
    failures: Sequence[ActionStageFailure],
) -> None:
    if failures:
        raise IsolatedActionStageError(stage, failures)
