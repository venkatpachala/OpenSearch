"""
Failure Taxonomy Classifier — rule-based first-pass failure classification.

This runs BEFORE the LLM to deterministically classify failures based on
observable facts (exit codes, metric ranges, constraint violations, etc.).

The LLM then gets a pre-classified failure type as part of its context,
making the evaluation more reliable and consistent.

Six failure types (from the architecture document):
  1. TOOL_CRASH           — exit code != 0 or exception
  2. TOOL_SCHEMA_ERROR    — Pydantic validation failure
  3. RESULT_INCONSISTENCY — training metric ≠ independent evaluation metric
  4. GOAL_DRIFT           — improves primary metric but violates constraint
  5. REGRESSION           — worse than current best
  6. NON_DETERMINISM      — high variance across identical runs
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..memory.models import FailureType


@dataclass
class TaxonomyResult:
    """Result of rule-based failure classification."""

    failure_type: FailureType | None
    """None if no failure is detected."""

    confidence: float
    """How confident is the rule-based classifier (0.0–1.0)."""

    reason: str
    """Human-readable explanation."""

    supporting_facts: dict[str, Any]
    """The specific facts that triggered this classification."""

    @property
    def is_failure(self) -> bool:
        return self.failure_type is not None


class FailureTaxonomyClassifier:
    """
    Rule-based failure classifier — runs without LLM.

    Rules are applied in priority order. The first matching rule wins.
    The classifier sets high confidence when the rule is unambiguous,
    lower confidence when the pattern is suggestive but not definitive.
    """

    def classify(
        self,
        tool_name: str,
        tool_response: dict[str, Any],
        goal_check_facts: dict[str, Any] | None = None,
    ) -> TaxonomyResult:
        """
        Apply classification rules in priority order.

        Args:
            tool_name:        The tool that produced this response.
            tool_response:    The raw tool response dict.
            goal_check_facts: Pre-computed GoalChecker facts (if available).
        """
        facts = goal_check_facts or {}

        # Rule 1: TOOL_SCHEMA_ERROR — schema error signal in response
        r = self._check_schema_error(tool_response)
        if r:
            return r

        # Explicit resource signals are distinct from generic tool crashes.
        r = self._check_resource_failure(tool_response)
        if r:
            return r

        # Rule 2: TOOL_CRASH — execution failure
        r = self._check_tool_crash(tool_response)
        if r:
            return r

        # Rule 3: RESULT_INCONSISTENCY — independent eval disagrees
        r = self._check_result_inconsistency(tool_response)
        if r:
            return r

        # Rule 4: GOAL_DRIFT — constraint violation despite metric improvement
        r = self._check_goal_drift(facts)
        if r:
            return r

        # Rule 5: REGRESSION — worse than best known
        r = self._check_regression(facts)
        if r:
            return r

        # Rule 6: NON_DETERMINISM — high variance detected
        r = self._check_non_determinism(facts)
        if r:
            return r

        # No failure detected
        return TaxonomyResult(
            failure_type=None,
            confidence=0.95,
            reason="No failure indicators detected",
            supporting_facts={},
        )

    # ── Rules ────────────────────────────────────────────────────────────────

    def _check_resource_failure(self, response: dict) -> TaxonomyResult | None:
        """Classify only explicit resource signals; legacy generic timeouts remain tool_crash."""
        data = response.get("data", {})
        if response.get("resource_failure") or data.get("resource_failure"):
            return TaxonomyResult(
                failure_type=FailureType.RESOURCE_FAILURE,
                confidence=0.98,
                reason="Execution reported an explicit resource exhaustion/unavailability signal",
                supporting_facts={"error": response.get("error"), "resource": data.get("resource")},
            )
        return None

    def _check_schema_error(self, response: dict) -> TaxonomyResult | None:
        error = response.get("error", "")
        data = response.get("data", {})
        error_l = error.lower() if isinstance(error, str) else ""
        argparse_schema = any(
            token in error_l
            for token in (
                "unrecognized arguments",
                "invalid choice",
                "the following arguments are required",
                "invalid int value",
                "invalid float value",
                "argument --",
            )
        ) and any(
            token in error_l
            for token in ("unrecognized", "invalid", "required", "expected", "argument")
        )
        if (
            error == "TOOL_SCHEMA_ERROR"
            or data.get("error_type") == "TOOL_SCHEMA_ERROR"
            or (isinstance(error, str) and "schema" in error_l)
            or argparse_schema
        ):
            return TaxonomyResult(
                failure_type=FailureType.TOOL_SCHEMA_ERROR,
                confidence=0.99,
                reason="Tool response contains schema error indicator",
                supporting_facts={
                    "error": error,
                    "field_errors": data.get("field_errors", []),
                },
            )
        return None

    def _check_tool_crash(self, response: dict) -> TaxonomyResult | None:
        if not response.get("success", True):
            error = response.get("error", "")
            timed_out = "timeout" in error.lower() or "timed out" in error.lower()
            crashed = "exit code" in error.lower() or "failed" in error.lower()

            if timed_out:
                return TaxonomyResult(
                    failure_type=FailureType.TOOL_CRASH,
                    confidence=0.99,
                    reason=f"Process timed out: {error[:100]}",
                    supporting_facts={"error": error, "timed_out": True},
                )
            if crashed or error:
                return TaxonomyResult(
                    failure_type=FailureType.TOOL_CRASH,
                    confidence=0.95,
                    reason=f"Tool execution failed: {error[:100]}",
                    supporting_facts={"error": error, "exit_code": response.get("exit_code")},
                )
        return None

    def _check_result_inconsistency(self, response: dict) -> TaxonomyResult | None:
        data = response.get("data", {})
        if "eval_accuracy" in data and "training_accuracy" in data:
            discrepancy = data.get("discrepancy", 0)
            consistent = data.get("consistent", True)
            if not consistent or discrepancy > 0.01:
                return TaxonomyResult(
                    failure_type=FailureType.RESULT_INCONSISTENCY,
                    confidence=0.95,
                    reason=(
                        f"Independent evaluation disagrees with training: "
                        f"discrepancy={discrepancy:.4f}"
                    ),
                    supporting_facts={
                        "eval_accuracy": data.get("eval_accuracy"),
                        "training_accuracy": data.get("training_accuracy"),
                        "discrepancy": discrepancy,
                    },
                )
        return None

    def _check_goal_drift(self, facts: dict) -> TaxonomyResult | None:
        if facts.get("any_constraint_violated") and facts.get("metric_achieved"):
            violated = [
                c for c in facts.get("constraint_details", [])
                if not c.get("satisfied")
            ]
            return TaxonomyResult(
                failure_type=FailureType.GOAL_DRIFT,
                confidence=0.90,
                reason=(
                    f"Metric achieved but constraint violated: "
                    + ", ".join(c["name"] for c in violated)
                ),
                supporting_facts={
                    "violated_constraints": violated,
                    "metric_achieved": True,
                },
            )
        return None

    def _check_regression(self, facts: dict) -> TaxonomyResult | None:
        if facts.get("is_regression"):
            delta = facts.get("regression_delta", 0)
            best = facts.get("best_known_value")
            current = facts.get("observed_value")
            return TaxonomyResult(
                failure_type=FailureType.REGRESSION,
                confidence=0.92,
                reason=(
                    f"Result regressed: current={current}, best={best}, "
                    f"delta={delta:.4f}"
                ),
                supporting_facts={
                    "current_value": current,
                    "best_known": best,
                    "delta": delta,
                },
            )
        return None

    def _check_non_determinism(self, facts: dict) -> TaxonomyResult | None:
        if facts.get("is_non_deterministic"):
            std = facts.get("std_dev")
            return TaxonomyResult(
                failure_type=FailureType.NON_DETERMINISM,
                confidence=0.85,
                reason=(
                    f"High result variance detected across runs: std_dev={std}"
                ),
                supporting_facts={"std_dev": std},
            )
        return None
