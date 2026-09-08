"""
Tests for EvaluationResult and related enumerations.
(No LLM calls — tests the Pydantic model directly.)
"""
import pytest

from research_repro.evaluation.evaluator import (
    ConstraintStatus,
    Decision,
    EvaluationResult,
    GoalProgress,
    HypothesisStatus,
)
from research_repro.memory.models import FailureType


def _make_result(**overrides) -> EvaluationResult:
    defaults = dict(
        execution_success=True,
        output_valid=True,
        metric_valid=True,
        goal_progress=GoalProgress.POSITIVE,
        constraint_status=ConstraintStatus.ALL_MET,
        hypothesis_status=HypothesisStatus.SUPPORTED,
        failure_type=None,
        decision=Decision.CONTINUE,
        confidence=0.9,
        rationale="All good.",
    )
    defaults.update(overrides)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def test_evaluation_result_success_case():
    result = _make_result()
    assert result.decision == Decision.CONTINUE
    assert result.failure_type is None
    assert result.confidence == 0.9


def test_evaluation_result_goal_achieved():
    result = _make_result(
        decision=Decision.GOAL_ACHIEVED,
        rationale="Accuracy 94.1% is within 0.5pp of target 94.2%",
    )
    assert result.decision == Decision.GOAL_ACHIEVED
    assert result.failure_type is None


def test_evaluation_result_discrepancy_failure():
    result = _make_result(
        execution_success=True,
        metric_valid=False,
        goal_progress=GoalProgress.NEGATIVE,
        hypothesis_status=HypothesisStatus.REJECTED,
        failure_type=FailureType.RESULT_INCONSISTENCY,
        decision=Decision.DIAGNOSE_AND_RECOVER,
        confidence=0.85,
        rationale="Observed 86.7% vs reported 94.2% — result inconsistency.",
    )
    assert result.decision == Decision.DIAGNOSE_AND_RECOVER
    assert result.failure_type == FailureType.RESULT_INCONSISTENCY
    assert result.goal_progress == GoalProgress.NEGATIVE
    assert result.hypothesis_status == HypothesisStatus.REJECTED


def test_evaluation_result_tool_crash():
    result = _make_result(
        execution_success=False,
        output_valid=False,
        metric_valid=False,
        goal_progress=GoalProgress.UNKNOWN,
        failure_type=FailureType.TOOL_CRASH,
        decision=Decision.DIAGNOSE_AND_RECOVER,
        confidence=0.99,
        rationale="Process exited with code 1.",
    )
    assert result.failure_type == FailureType.TOOL_CRASH
    assert result.execution_success is False


def test_evaluation_result_goal_drift():
    result = _make_result(
        constraint_status=ConstraintStatus.VIOLATED,
        failure_type=FailureType.GOAL_DRIFT,
        decision=Decision.DIAGNOSE_AND_RECOVER,
        rationale="Accuracy improved but latency constraint violated.",
    )
    assert result.failure_type == FailureType.GOAL_DRIFT
    assert result.constraint_status == ConstraintStatus.VIOLATED


def test_evaluation_result_regression():
    result = _make_result(
        goal_progress=GoalProgress.NEGATIVE,
        hypothesis_status=HypothesisStatus.REJECTED,
        failure_type=FailureType.REGRESSION,
        decision=Decision.DIAGNOSE_AND_RECOVER,
        rationale="F1 dropped from 0.91 to 0.87.",
    )
    assert result.failure_type == FailureType.REGRESSION


def test_evaluation_result_non_determinism():
    result = _make_result(
        failure_type=FailureType.NON_DETERMINISM,
        decision=Decision.DIAGNOSE_AND_RECOVER,
        rationale="Runs 1-3: 91.2%, 87.3%, 90.8% — high variance detected.",
    )
    assert result.failure_type == FailureType.NON_DETERMINISM


def test_evaluation_result_schema_error():
    result = _make_result(
        execution_success=False,
        failure_type=FailureType.TOOL_SCHEMA_ERROR,
        decision=Decision.DIAGNOSE_AND_RECOVER,
        rationale="experiment_id must be str, received int.",
    )
    assert result.failure_type == FailureType.TOOL_SCHEMA_ERROR


# ---------------------------------------------------------------------------
# Enum coverage — all 6 failure types must exist
# ---------------------------------------------------------------------------


def test_all_six_failure_types_exist():
    types = {ft.value for ft in FailureType}
    assert types == {
        "tool_crash",
        "tool_schema_error",
        "result_inconsistency",
        "goal_drift",
        "regression",
        "non_determinism",
    }


def test_all_decisions_exist():
    decisions = {d.value for d in Decision}
    assert decisions == {"continue", "diagnose_and_recover", "goal_achieved", "terminate"}


def test_all_goal_progress_values():
    values = {g.value for g in GoalProgress}
    assert values == {"positive", "negative", "neutral", "unknown"}
