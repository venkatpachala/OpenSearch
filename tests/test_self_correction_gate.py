"""Evaluate-before-commit and timeout-retry gates for self-correction."""
from __future__ import annotations

from research_repro.agent.loop import AgentLoop
from research_repro.agent.react import StepResult
from research_repro.agent.planner import ReasoningTrace
from research_repro.agent.strategy import SelfCorrectingStrategy
from research_repro.config import Config
from research_repro.evaluation.evaluator import (
    ConstraintStatus,
    Decision,
    EvaluationResult,
    GoalProgress,
    HypothesisStatus,
)
from research_repro.memory.models import Constraint, Failure, FailureType, GoalContract, MetricCriterion
from research_repro.recovery import RecoveryOrchestrator
from research_repro.recovery.strategies import next_experiment_id
from research_repro.tools.base import ToolResponse
from research_repro.tools.stub import build_stub_registry


def _goal() -> GoalContract:
    return GoalContract(
        objective="Improve MNIST accuracy to at least 96% with latency under 100ms",
        primary_metric="accuracy",
        target_value=0.96,
        success_threshold=0.0,
        criterion=MetricCriterion(metric="accuracy", direction="maximize", target=0.96, tolerance=0.0, minimum=0.96),
        constraints=[Constraint(name="latency_under_100ms", description="latency_under_100ms")],
    )


def _eval(**kwargs) -> EvaluationResult:
    defaults = dict(
        execution_success=True,
        output_valid=True,
        metric_valid=True,
        goal_progress=GoalProgress.POSITIVE,
        constraint_status=ConstraintStatus.ALL_MET,
        hypothesis_status=HypothesisStatus.INCONCLUSIVE,
        failure_type=None,
        decision=Decision.CONTINUE,
        confidence=0.9,
        rationale="test",
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


def _step(tmp_loop, evaluation: EvaluationResult, data: dict, tool_args: dict | None = None) -> StepResult:
    return StepResult(
        step_index=1,
        reasoning=ReasoningTrace(
            goal_relevance="g", evidence_basis="e", hypothesis="h", intended_action="run",
        ),
        tool_name="run_experiment",
        tool_args=tool_args or {"experiment_id": "exp_01", "parameters": {"normalize": True}},
        tool_response=ToolResponse.ok(**data),
        evaluation=evaluation,
        decision=evaluation.decision,
    )


def test_next_experiment_id_increments():
    assert next_experiment_id("exp_01") == "exp_02"
    assert next_experiment_id("exp_9") == "exp_10"


def test_metric_hit_with_latency_violation_is_not_contract_pass():
    goal = _goal()
    assert goal.is_achieved(0.9635)
    assert not goal.constraints_hold({"latency_ms": 156.9})
    assert not goal.meets_contract(0.9635, {"latency_ms": 156.9})
    assert goal.meets_contract(0.9635, {"latency_ms": 80.0})


def test_drift_run_is_recorded_but_not_best(tmp_path):
    loop = AgentLoop(_goal(), "gate_drift", Config(runs_dir=tmp_path), build_stub_registry())
    result = _step(
        loop,
        _eval(
            constraint_status=ConstraintStatus.VIOLATED,
            failure_type=FailureType.GOAL_DRIFT,
            decision=Decision.DIAGNOSE_AND_RECOVER,
            rationale="accuracy on target but latency_under_100ms violated",
        ),
        {"experiment_id": "exp_01", "accuracy": 0.9635, "latency_ms": 156.9, "f1": 0.9635, "loss": 0.001},
    )
    loop._auto_advance_phase(result)
    assert result.decision == Decision.DIAGNOSE_AND_RECOVER
    loop._handle_experiment_result(result)
    assert loop.memory.experiments[0].id == "exp_01"
    assert loop.memory.experiments[0].evaluation_status.value == "negative"
    assert loop.memory.best_experiment_id is None


def test_update_best_ignores_constraint_violating_runs(tmp_path):
    loop = AgentLoop(_goal(), "gate_best", Config(runs_dir=tmp_path), build_stub_registry())
    drifted = _step(
        loop,
        _eval(
            constraint_status=ConstraintStatus.VIOLATED,
            failure_type=FailureType.GOAL_DRIFT,
            decision=Decision.DIAGNOSE_AND_RECOVER,
        ),
        {"experiment_id": "exp_01", "accuracy": 0.961, "latency_ms": 156.9},
    )
    legal = _step(
        loop,
        _eval(decision=Decision.CONTINUE, goal_progress=GoalProgress.NEGATIVE),
        {"experiment_id": "exp_04", "accuracy": 0.932, "latency_ms": 76.1},
        tool_args={"experiment_id": "exp_04", "parameters": {"hidden_size": 64}},
    )
    loop._handle_experiment_result(drifted)
    loop._handle_experiment_result(legal)
    assert loop.memory.best_experiment_id == "exp_04"


def test_legal_run_can_become_best(tmp_path):
    loop = AgentLoop(_goal(), "gate_legal", Config(runs_dir=tmp_path), build_stub_registry())
    result = _step(
        loop,
        _eval(decision=Decision.GOAL_ACHIEVED, hypothesis_status=HypothesisStatus.SUPPORTED),
        {"experiment_id": "exp_02", "accuracy": 0.961, "latency_ms": 76.0},
        tool_args={"experiment_id": "exp_02", "parameters": {"hidden_size": 64, "hidden_layers": 1, "normalize": True}},
    )
    loop._handle_experiment_result(result)
    assert loop.memory.best_experiment_id == "exp_02"


def test_compare_results_reproduced_does_not_close_the_run(tmp_path):
    loop = AgentLoop(_goal(), "gate_compare", Config(runs_dir=tmp_path), build_stub_registry())
    result = StepResult(
        step_index=2,
        reasoning=ReasoningTrace(goal_relevance="g", evidence_basis="e", hypothesis="h", intended_action="compare"),
        tool_name="compare_results",
        tool_args={"experiment_id": "exp_01", "reported_value": 0.96, "observed_value": 0.961},
        tool_response=ToolResponse.ok(status="REPRODUCED", reported=0.96, observed=0.961),
        evaluation=_eval(decision=Decision.DIAGNOSE_AND_RECOVER, failure_type=FailureType.GOAL_DRIFT),
        decision=Decision.DIAGNOSE_AND_RECOVER,
    )
    loop._auto_advance_phase(result)
    assert result.decision == Decision.DIAGNOSE_AND_RECOVER


def test_timeout_strategy_retries_new_id_not_same_payload():
    memory_goal = _goal()
    from research_repro.memory.models import ResearchMemory
    memory = ResearchMemory(goal=memory_goal, run_id="gate_timeout")
    memory.failures.append(Failure(failure_type=FailureType.TOOL_CRASH, description="timed out", experiment_id="exp_01"))

    class Eval:
        failure_type = FailureType.TOOL_CRASH

    class Response:
        def model_dump(self, mode="json"):
            return {"success": False, "error": "Experiment timed out after 300s. Partial stdout: Epoch 1/200"}

    class Result:
        step_index = 1
        tool_name = "run_experiment"
        tool_args = {"experiment_id": "exp_01", "timeout_seconds": 300, "parameters": {"max_iter": 100, "normalize": True}}
        evaluation = Eval()
        tool_response = Response()

    decision = SelfCorrectingStrategy(RecoveryOrchestrator()).handle_result(Result(), memory)
    assert decision.action == "retry"
    assert decision.retry_same_action is False
    assert decision.repaired_tool_args["experiment_id"] == "exp_02"
    assert decision.repaired_tool_args["parameters"]["max_iter"] == 50
