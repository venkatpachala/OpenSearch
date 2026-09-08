"""Causally targeted recovery strategies for all six failure classes."""
from __future__ import annotations

import copy
import math
import statistics
import time
from typing import Any

from ..memory.models import FailureType
from .base import RecoveryContext, RecoveryOutcome, RecoveryStrategy


class ToolCrashRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.TOOL_CRASH
    name = "tool_crash_repair_and_retry"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        args = copy.deepcopy(context.tool_args)
        error = str(context.tool_response.get("error", "")).lower()
        params = args.setdefault("parameters", {})
        if "out of memory" in error or "cuda" in error:
            if isinstance(params.get("batch_size"), int) and params["batch_size"] > 1:
                params["batch_size"] = max(1, params["batch_size"] // 2)
            else:
                params["gradient_checkpointing"] = True
            action = "Reduced batch_size or enabled gradient checkpointing after CUDA failure"
        elif "timeout" in error or context.tool_response.get("timed_out"):
            args["timeout_seconds"] = min(86400, int(args.get("timeout_seconds", 300) * 1.5))
            if isinstance(params.get("epochs"), int) and params["epochs"] > 1:
                params["epochs"] = max(1, params["epochs"] // 2)
            action = "Adjusted timeout/iteration budget after process timeout"
        else:
            action = "Retried transient tool failure with bounded backoff"
        time.sleep(0.01)
        return RecoveryOutcome(True, self.name, action, {"tool_args": args}, retry=True)


class SchemaRepairStrategy(RecoveryStrategy):
    failure_type = FailureType.TOOL_SCHEMA_ERROR
    name = "schema_validate_and_repair"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        errors = context.tool_response.get("data", {}).get("field_errors", [])
        if context.schema_repair is None:
            return RecoveryOutcome(False, self.name, "No typed schema repair callback configured", unresolved="schema repair unavailable")
        repaired = context.schema_repair(context.tool_args, errors)
        return RecoveryOutcome(True, self.name, "Repaired tool arguments from Pydantic field feedback", {"tool_args": repaired}, retry=True)


class ResultInconsistencyRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.RESULT_INCONSISTENCY
    name = "independent_metric_reaudit"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        if context.independent_evaluator is None:
            return RecoveryOutcome(False, self.name, "Independent evaluator is not configured", unresolved="metric could not be independently re-audited")
        audit = context.independent_evaluator(context.tool_args)
        return RecoveryOutcome(True, self.name, "Re-audited metric with independent evaluator", {"audit": audit}, retry=False)


class GoalDriftRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.GOAL_DRIFT
    name = "constraint_reject_and_rollback"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        commit = context.memory.get_best_result() and next((e.git_commit for e in context.memory.experiments if e.id == context.memory.best_experiment_id), None)
        restored = bool(commit and context.checkpointer and context.checkpointer.restore(commit))
        action = "Rejected constraint-violating result"
        if restored:
            action += f" and restored compliant checkpoint {commit}"
        return RecoveryOutcome(restored or not commit, self.name, action, {"constraint_violation": context.tool_response})


class RegressionRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.REGRESSION
    name = "best_node_rollback_and_prune"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        best = max((e for e in context.memory.experiments if e.observed_result), key=lambda e: e.observed_result.get_metric(context.memory.goal.primary_metric) or float("-inf"), default=None)
        restored = bool(best and best.git_commit and context.checkpointer and context.checkpointer.restore(best.git_commit))
        return RecoveryOutcome(True, self.name, "Rejected regression and restored best known experiment" if restored else "Rejected regression; preserved best known experiment", {"best_experiment_id": best.id if best else None})


class NonDeterminismRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.NON_DETERMINISM
    name = "multi_seed_variance_filter"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        samples = context.tool_response.get("samples", [])
        if not samples:
            samples = [e.observed_result.get_metric(context.memory.goal.primary_metric) for e in context.memory.experiments if e.observed_result and e.observed_result.get_metric(context.memory.goal.primary_metric) is not None]
        samples = [float(x) for x in samples if isinstance(x, (int, float))]
        if len(samples) < 2:
            return RecoveryOutcome(False, self.name, "Insufficient repeated samples for variance estimate", unresolved="non-determinism requires repeated seeds")
        mean = statistics.mean(samples)
        std = statistics.stdev(samples)
        se = std / math.sqrt(len(samples))
        return RecoveryOutcome(True, self.name, "Computed multi-sample mean and conservative 95% lower bound", {"mean": mean, "std": std, "standard_error": se, "lower_bound_95": mean - 1.96 * se})


STRATEGIES = {s.failure_type: s for s in (
    ToolCrashRecoveryStrategy(), SchemaRepairStrategy(), ResultInconsistencyRecoveryStrategy(),
    GoalDriftRecoveryStrategy(), RegressionRecoveryStrategy(), NonDeterminismRecoveryStrategy(),
)}

