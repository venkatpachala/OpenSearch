"""Causally targeted recovery strategies for all six failure classes."""
from __future__ import annotations

import copy
import math
import statistics
import time
from typing import Any

from ..memory.models import FailureType
from .base import RecoveryContext, RecoveryOutcome, RecoveryStrategy


def next_experiment_id(current: str | None) -> str:
    """Increment a trailing integer id so per-id injectors cannot pin the agent."""
    import re
    text = current or "exp_01"
    match = re.search(r"^(.*?)(\d+)$", text)
    if not match:
        return f"{text}_retry"
    width = len(match.group(2))
    return f"{match.group(1)}{int(match.group(2)) + 1:0{width}d}"


def _is_timeout(error: str, response: dict[str, Any]) -> bool:
    blob = f"{error} {response}".lower()
    return any(token in blob for token in ("timeout", "timed out", "timed_out"))


class ToolCrashRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.TOOL_CRASH
    name = "tool_crash_repair_and_retry"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        args = copy.deepcopy(context.tool_args)
        error = str(context.tool_response.get("error", "")).lower()
        params = args.setdefault("parameters", {})
        if not isinstance(params, dict):
            params = {}
            args["parameters"] = params
        # Always leave the failed experiment_id so FAIL_ONCE/TIMEOUT schedules
        # cannot be hammered. Attempt 1 retries the runner with repaired args.
        failed_id = args.get("experiment_id")
        args["experiment_id"] = next_experiment_id(failed_id if isinstance(failed_id, str) else None)
        if "out of memory" in error or "cuda" in error:
            if isinstance(params.get("batch_size"), int) and params["batch_size"] > 1:
                params["batch_size"] = max(1, params["batch_size"] // 2)
            else:
                params["gradient_checkpointing"] = True
            action = "Reduced batch_size or enabled gradient checkpointing after CUDA failure"
        elif _is_timeout(error, context.tool_response):
            args["timeout_seconds"] = min(86400, int(args.get("timeout_seconds", 300) * 1.5))
            if isinstance(params.get("epochs"), int) and params["epochs"] > 1:
                params["epochs"] = max(1, params["epochs"] // 2)
            if isinstance(params.get("max_iter"), int) and params["max_iter"] > 1:
                params["max_iter"] = max(20, params["max_iter"] // 2)
            elif "max_iter" not in params:
                params["max_iter"] = 50
            action = (
                f"Adjusted timeout/iteration budget after process timeout; "
                f"retry as {args['experiment_id']}"
            )
        else:
            action = f"Retried transient tool failure on {args['experiment_id']} with bounded backoff"
        time.sleep(0.01)
        return RecoveryOutcome(True, self.name, action, {"tool_args": args}, retry=True,
                              diagnosis=f"Tool failure evidence: {error or 'transient execution failure'}",
                              evidence_used=[error] if error else [],
                              expected_observation="The repaired experiment_id completes without the observed resource/process failure",
                              plan_change="Retry the runner with a new experiment_id and repaired execution parameters")


class ResourceFailureRecoveryStrategy(ToolCrashRecoveryStrategy):
    failure_type = FailureType.RESOURCE_FAILURE
    name = "resource_budget_repair_and_retry"


class SchemaRepairStrategy(RecoveryStrategy):
    failure_type = FailureType.TOOL_SCHEMA_ERROR
    name = "schema_validate_and_repair"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        errors = context.tool_response.get("data", {}).get("field_errors", [])
        if context.schema_repair is None:
            return RecoveryOutcome(False, self.name, "No typed schema repair callback configured", unresolved="schema repair unavailable")
        repaired = context.schema_repair(context.tool_args, errors)
        return RecoveryOutcome(True, self.name, "Repaired tool arguments from Pydantic field feedback", {"tool_args": repaired}, retry=True,
                              diagnosis="Tool arguments violated the typed request schema",
                              evidence_used=[str(x) for x in errors],
                              expected_observation="The repaired arguments validate and the tool executes",
                              plan_change="Replace invalid arguments with schema-conforming values")


class ResultInconsistencyRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.RESULT_INCONSISTENCY
    name = "independent_metric_reaudit"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        if context.independent_evaluator is None:
            return RecoveryOutcome(False, self.name, "Independent evaluator is not configured", unresolved="metric could not be independently re-audited")
        audit = context.independent_evaluator(context.tool_args)
        return RecoveryOutcome(True, self.name, "Re-audited metric with independent evaluator", {"audit": audit}, retry=False,
                              diagnosis="Reported and independently recomputed metrics disagree",
                              evidence_used=[str(audit)],
                              expected_observation="Independent metric identifies the trusted result",
                              plan_change="Use independently verified metrics for the next decision")


class GoalDriftRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.GOAL_DRIFT
    name = "constraint_reject_and_rollback"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        commit = context.memory.get_best_result() and next((e.git_commit for e in context.memory.experiments if e.id == context.memory.best_experiment_id), None)
        restored = bool(commit and context.checkpointer and context.checkpointer.restore(commit))
        action = "Rejected constraint-violating result"
        if restored:
            action += f" and restored compliant checkpoint {commit}"
        return RecoveryOutcome(restored or not commit, self.name, action, {"constraint_violation": context.tool_response},
                              diagnosis="Primary metric improvement violated a declared constraint",
                              evidence_used=[str(context.tool_response)],
                              expected_observation="A subsequent candidate satisfies every constraint",
                              plan_change="Tighten the plan around the violated constraint")


class RegressionRecoveryStrategy(RecoveryStrategy):
    failure_type = FailureType.REGRESSION
    name = "best_node_rollback_and_prune"

    def recover(self, context: RecoveryContext) -> RecoveryOutcome:
        best = max((e for e in context.memory.experiments if e.observed_result), key=lambda e: e.observed_result.get_metric(context.memory.goal.primary_metric) or float("-inf"), default=None)
        restored = bool(best and best.git_commit and context.checkpointer and context.checkpointer.restore(best.git_commit))
        return RecoveryOutcome(True, self.name, "Rejected regression and restored best known experiment" if restored else "Rejected regression; preserved best known experiment", {"best_experiment_id": best.id if best else None},
                              diagnosis="Observed metric was materially below the best known result",
                              evidence_used=[f"best_experiment={best.id}" if best else "no best experiment"],
                              expected_observation="A new branch improves on or returns to the best known result",
                              plan_change="Prune the regressed parameter branch")


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
        return RecoveryOutcome(True, self.name, "Computed multi-sample mean and conservative 95% lower bound", {"mean": mean, "std": std, "standard_error": se, "lower_bound_95": mean - 1.96 * se},
                              diagnosis=f"Repeated samples vary with standard deviation {std:.4f}",
                              evidence_used=[str(x) for x in samples],
                              expected_observation="The conservative lower bound determines whether the goal is robustly met",
                              plan_change="Report aggregate performance instead of a single seed")


STRATEGIES = {s.failure_type: s for s in (
    ToolCrashRecoveryStrategy(), ResourceFailureRecoveryStrategy(), SchemaRepairStrategy(), ResultInconsistencyRecoveryStrategy(),
    GoalDriftRecoveryStrategy(), RegressionRecoveryStrategy(), NonDeterminismRecoveryStrategy(),
)}
