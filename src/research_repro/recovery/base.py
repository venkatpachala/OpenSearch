"""Typed recovery interfaces.

Recovery is deliberately expressed as operations on the real runner and
workspace.  Strategies never execute arbitrary commands or invent metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from ..memory.models import Failure, FailureType, ResearchMemory


@dataclass
class RecoveryContext:
    memory: ResearchMemory
    failure: Failure
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_response: dict[str, Any] = field(default_factory=dict)
    runner: Any | None = None
    experiment_config: Any | None = None
    checkpointer: Any | None = None
    independent_evaluator: Callable[..., dict[str, Any]] | None = None
    schema_repair: Callable[[dict[str, Any], list[dict[str, Any]]], dict[str, Any]] | None = None


@dataclass
class RecoveryOutcome:
    success: bool
    strategy: str
    action_taken: str
    updates: dict[str, Any] = field(default_factory=dict)
    retry: bool = False
    unresolved: str | None = None


class RecoveryStrategy:
    failure_type: FailureType
    name: str

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        raise NotImplementedError

