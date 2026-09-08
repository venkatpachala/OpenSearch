"""
Tests for Phase 6: Recovery strategies and RecoveryOrchestrator.
Covers all 6 causally distinct failure classes and budget enforcement.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from research_repro.memory.models import (
    Failure, FailureType, GoalContract, ResearchMemory, Experiment, ExperimentResult, ResourceBudget
)
from research_repro.recovery.base import RecoveryContext
from research_repro.recovery.orchestrator import RecoveryOrchestrator
from research_repro.recovery.strategies import (
    ToolCrashRecoveryStrategy, SchemaRepairStrategy,
    ResultInconsistencyRecoveryStrategy, GoalDriftRecoveryStrategy,
    RegressionRecoveryStrategy, NonDeterminismRecoveryStrategy,
)
from research_repro.experiments.checkpoints import GitCheckpointer


def _make_memory() -> ResearchMemory:
    goal = GoalContract(
        objective="Reproduce benchmark experiment",
        primary_metric="accuracy",
        target_value=0.95,
        allowed_resources=ResourceBudget(max_experiments=8, max_recoveries_per_subtask=2, max_total_recoveries=6),
    )
    return ResearchMemory(goal=goal, run_id="test_recovery_run")


class TestRecoveryStrategies:
    def test_tool_crash_oom_halves_batch_size(self):
        strat = ToolCrashRecoveryStrategy()
        mem = _make_memory()
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.TOOL_CRASH, description="CUDA out of memory", step_index=1),
            tool_args={"parameters": {"batch_size": 64}},
            tool_response={"error": "CUDA out of memory while allocating tensor"},
        )
        outcome = strat.recover(ctx)
        assert outcome.success is True
        assert outcome.retry is True
        assert outcome.updates["tool_args"]["parameters"]["batch_size"] == 32

    def test_tool_crash_timeout_increases_budget(self):
        strat = ToolCrashRecoveryStrategy()
        mem = _make_memory()
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.TOOL_CRASH, description="Process timeout", step_index=1),
            tool_args={"timeout_seconds": 100, "parameters": {"epochs": 20}},
            tool_response={"timed_out": True, "error": "Process timed out after 100s"},
        )
        outcome = strat.recover(ctx)
        assert outcome.success is True
        assert outcome.updates["tool_args"]["timeout_seconds"] == 150
        assert outcome.updates["tool_args"]["parameters"]["epochs"] == 10

    def test_schema_repair_invokes_callback(self):
        strat = SchemaRepairStrategy()
        mem = _make_memory()
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.TOOL_SCHEMA_ERROR, description="Schema error", step_index=1),
            tool_args={"seed": "forty-two"},
            tool_response={"data": {"field_errors": [{"field": "seed", "type": "int_parsing"}]}},
            schema_repair=lambda args, errors: {"seed": 42},
        )
        outcome = strat.recover(ctx)
        assert outcome.success is True
        assert outcome.updates["tool_args"]["seed"] == 42

    def test_result_inconsistency_reaudits_metric(self):
        strat = ResultInconsistencyRecoveryStrategy()
        mem = _make_memory()
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.RESULT_INCONSISTENCY, description="Inconsistent metrics", step_index=1),
            tool_args={"experiment_id": "exp_001"},
            tool_response={"discrepancy": 0.04},
            independent_evaluator=lambda args: {"eval_accuracy": 0.912, "verified": True},
        )
        outcome = strat.recover(ctx)
        assert outcome.success is True
        assert outcome.updates["audit"]["eval_accuracy"] == 0.912

    def test_goal_drift_rejects_and_restores_checkpoint(self, tmp_path):
        checkpointer = GitCheckpointer(workspace_dir=tmp_path)
        checkpointer.initialize()
        (tmp_path / "model.txt").write_text("v1")
        c1 = checkpointer.checkpoint("exp_001", "good commit")

        mem = _make_memory()
        exp = Experiment(id="exp_001", git_commit=c1)
        exp.observed_result = ExperimentResult(accuracy=0.94)
        mem.experiments.append(exp)
        mem.best_experiment_id = "exp_001"

        # Mutate file to simulate drift
        (tmp_path / "model.txt").write_text("v2_drifted")

        strat = GoalDriftRecoveryStrategy()
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.GOAL_DRIFT, description="Latency exceeded", step_index=2),
            tool_args={"parameters": {"layers": 10}},
            tool_response={"latency_ms": 150},
            checkpointer=checkpointer,
        )
        outcome = strat.recover(ctx)
        assert outcome.success is True
        assert "Rejected constraint-violating result" in outcome.action_taken
        assert (tmp_path / "model.txt").read_text() == "v1"

    def test_regression_restores_best_experiment(self, tmp_path):
        checkpointer = GitCheckpointer(workspace_dir=tmp_path)
        checkpointer.initialize()
        (tmp_path / "cfg.json").write_text('{"best": true}')
        c_best = checkpointer.checkpoint("exp_001", "best run")

        mem = _make_memory()
        exp_best = Experiment(id="exp_001", git_commit=c_best)
        exp_best.observed_result = ExperimentResult(accuracy=0.96)

        exp_regressed = Experiment(id="exp_002")
        exp_regressed.observed_result = ExperimentResult(accuracy=0.85)

        mem.experiments.extend([exp_best, exp_regressed])

        (tmp_path / "cfg.json").write_text('{"best": false}')

        strat = RegressionRecoveryStrategy()
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.REGRESSION, description="Regression detected", step_index=2),
            checkpointer=checkpointer,
        )
        outcome = strat.recover(ctx)
        assert outcome.success is True
        assert (tmp_path / "cfg.json").read_text() == '{"best": true}'

    def test_non_determinism_computes_stats(self):
        strat = NonDeterminismRecoveryStrategy()
        mem = _make_memory()
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.NON_DETERMINISM, description="Variance detected", step_index=1),
            tool_response={"samples": [0.91, 0.93, 0.95]},
        )
        outcome = strat.recover(ctx)
        assert outcome.success is True
        assert outcome.updates["mean"] == pytest.approx(0.93)
        assert outcome.updates["lower_bound_95"] < outcome.updates["mean"]


class TestRecoveryOrchestrator:
    def test_orchestrator_enforces_subtask_limit(self):
        mem = _make_memory()
        orch = RecoveryOrchestrator(max_total_recoveries=6, max_recoveries_per_subtask=2)
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.TOOL_CRASH, description="fail", step_index=1),
            tool_args={"parameters": {}},
            tool_response={"error": "transient"},
        )

        # Attempt 1
        o1 = orch.recover(ctx, subtask_id="train_step")
        assert o1.success is True
        assert mem.budget.recoveries_consumed == 1

        # Attempt 2
        o2 = orch.recover(ctx, subtask_id="train_step")
        assert o2.success is True
        assert mem.budget.recoveries_consumed == 2

        # Attempt 3 — should hit subtask limit
        o3 = orch.recover(ctx, subtask_id="train_step")
        assert o3.success is False
        assert o3.strategy == "budget_exhausted"
        assert "unresolvable subtask" in o3.unresolved

    def test_orchestrator_enforces_total_limit(self):
        mem = _make_memory()
        orch = RecoveryOrchestrator(max_total_recoveries=2, max_recoveries_per_subtask=5)
        ctx = RecoveryContext(
            memory=mem,
            failure=Failure(failure_type=FailureType.TOOL_CRASH, description="fail", step_index=1),
            tool_args={"parameters": {}},
            tool_response={"error": "transient"},
        )

        orch.recover(ctx, subtask_id="subtask_1")
        orch.recover(ctx, subtask_id="subtask_2")

        # Third total recovery exceeds max_total_recoveries=2
        o3 = orch.recover(ctx, subtask_id="subtask_3")
        assert o3.success is False
        assert o3.strategy == "budget_exhausted"
