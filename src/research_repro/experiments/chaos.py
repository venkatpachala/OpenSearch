"""
Phase 8: Chaos Engineering & Fault Injection Testbed.

Systematically stress-tests the agent under adversarial failure modes
to verify that self-correction guarantees hold across all 6 failure types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..memory.models import FailureType
from .fault_injection import FaultMode, FaultInjectingRunner


@dataclass
class ChaosFault:
    id: str
    failure_type: FailureType
    expected_strategy: str
    fault_mode: FaultMode = FaultMode.NORMAL
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "failure_type": self.failure_type.value,
            "expected_strategy": self.expected_strategy,
            "fault_mode": self.fault_mode.value,
            "description": self.description,
        }


# The 6 canonical chaos evaluation cases matching all 6 failure classes
FAULT_MATRIX: list[ChaosFault] = [
    ChaosFault(
        id="case_tool_crash",
        failure_type=FailureType.TOOL_CRASH,
        expected_strategy="tool_crash_repair_and_retry",
        fault_mode=FaultMode.FAIL_ONCE,
        description="Subprocess failure / crash",
    ),
    ChaosFault(
        id="case_schema_error",
        failure_type=FailureType.TOOL_SCHEMA_ERROR,
        expected_strategy="schema_validate_and_repair",
        description="Invalid parameter schema",
    ),
    ChaosFault(
        id="case_result_inconsistency",
        failure_type=FailureType.RESULT_INCONSISTENCY,
        expected_strategy="independent_metric_reaudit",
        fault_mode=FaultMode.CORRUPT_METRIC,
        description="Divergent metric between training and evaluation",
    ),
    ChaosFault(
        id="case_goal_drift",
        failure_type=FailureType.GOAL_DRIFT,
        expected_strategy="constraint_reject_and_rollback",
        fault_mode=FaultMode.WRONG_RESULT,
        description="Constraint breach despite metric improvement",
    ),
    ChaosFault(
        id="case_regression",
        failure_type=FailureType.REGRESSION,
        expected_strategy="best_node_rollback_and_prune",
        description="Experiment performs worse than best historical node",
    ),
    ChaosFault(
        id="case_non_determinism",
        failure_type=FailureType.NON_DETERMINISM,
        expected_strategy="multi_seed_variance_filter",
        description="High seed-to-seed variance across identical parameters",
    ),
]


class ChaosEngine:
    """Injects and controls adversarial failure modes during experiment execution."""

    def __init__(self, matrix: list[ChaosFault] | None = None):
        self.matrix = matrix or FAULT_MATRIX

    def get_fault_for(self, failure_type: FailureType) -> ChaosFault | None:
        return next((f for f in self.matrix if f.failure_type == failure_type), None)
