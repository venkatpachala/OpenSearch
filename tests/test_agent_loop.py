"""
Tests for core agent components:
  - StateMachine transitions
  - BudgetState logic
  - GoalContract achievement check
  - ResearchMemory helpers
  - ExperimentGraph
  - MemoryStore round-trip
"""
import json
import tempfile
from pathlib import Path

import pytest

from research_repro.agent.state_machine import (
    AgentPhase,
    InvalidTransitionError,
    StateMachine,
)
from research_repro.memory.experiment_graph import ExperimentGraph
from research_repro.memory.models import (
    BudgetState,
    Constraint,
    Experiment,
    ExperimentResult,
    GoalContract,
    Plan,
    ResearchMemory,
)
from research_repro.memory.store import MemoryStore


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class TestStateMachine:
    def test_initial_phase(self):
        sm = StateMachine()
        assert sm.current == AgentPhase.INITIALIZING

    def test_valid_transition(self):
        sm = StateMachine()
        sm.transition(AgentPhase.LITERATURE_DISCOVERY)
        assert sm.current == AgentPhase.LITERATURE_DISCOVERY

    def test_valid_chain(self):
        sm = StateMachine()
        sm.transition(AgentPhase.LITERATURE_DISCOVERY)
        sm.transition(AgentPhase.PAPER_SELECTION)
        sm.transition(AgentPhase.METHOD_EXTRACTION)
        sm.transition(AgentPhase.EXPERIMENT_PLANNING)
        assert sm.current == AgentPhase.EXPERIMENT_PLANNING

    def test_invalid_transition_raises(self):
        sm = StateMachine()
        with pytest.raises(InvalidTransitionError) as exc_info:
            sm.transition(AgentPhase.COMPLETED)
        assert "INITIALIZING" in str(exc_info.value).upper() or "initializing" in str(exc_info.value)

    def test_invalid_transition_message_contains_valid_targets(self):
        sm = StateMachine()
        try:
            sm.transition(AgentPhase.GENERATING_REPORT)
        except InvalidTransitionError as e:
            assert "literature_discovery" in str(e)

    def test_terminal_phases(self):
        sm_done = StateMachine(AgentPhase.COMPLETED)
        assert sm_done.is_terminal is True

        sm_fail = StateMachine(AgentPhase.FAILED)
        assert sm_fail.is_terminal is True

    def test_non_terminal_phase(self):
        sm = StateMachine(AgentPhase.RUNNING_EXPERIMENT)
        assert sm.is_terminal is False

    def test_can_transition_check(self):
        sm = StateMachine()
        assert sm.can_transition(AgentPhase.LITERATURE_DISCOVERY) is True
        assert sm.can_transition(AgentPhase.COMPLETED) is False

    def test_self_evaluation_can_go_to_multiple_targets(self):
        sm = StateMachine(AgentPhase.SELF_EVALUATION)
        assert sm.can_transition(AgentPhase.RUNNING_EXPERIMENT)
        assert sm.can_transition(AgentPhase.DIAGNOSING)
        assert sm.can_transition(AgentPhase.GENERATING_REPORT)


# ---------------------------------------------------------------------------
# Budget state
# ---------------------------------------------------------------------------


class TestBudgetState:
    def test_can_run_experiment_within_budget(self):
        budget = BudgetState(experiments_consumed=5)
        assert budget.can_run_experiment(8) is True

    def test_cannot_run_experiment_at_limit(self):
        budget = BudgetState(experiments_consumed=8)
        assert budget.can_run_experiment(8) is False

    def test_can_recover_within_budget(self):
        budget = BudgetState(recoveries_consumed=3)
        assert budget.can_recover(6) is True

    def test_cannot_recover_at_limit(self):
        budget = BudgetState(recoveries_consumed=6)
        assert budget.can_recover(6) is False

    def test_consume_experiment_increments(self):
        budget = BudgetState()
        budget.consume_experiment()
        budget.consume_experiment()
        assert budget.experiments_consumed == 2

    def test_consume_recovery_increments(self):
        budget = BudgetState()
        budget.consume_recovery("task_1")
        budget.consume_recovery("task_1")
        assert budget.recoveries_consumed == 2
        assert budget.recoveries_per_subtask["task_1"] == 2

    def test_can_recover_subtask(self):
        budget = BudgetState()
        budget.recoveries_per_subtask["t1"] = 1
        assert budget.can_recover_subtask("t1", 2) is True
        budget.recoveries_per_subtask["t1"] = 2
        assert budget.can_recover_subtask("t1", 2) is False

    def test_new_subtask_starts_at_zero(self):
        budget = BudgetState()
        assert budget.can_recover_subtask("new_task", 2) is True


# ---------------------------------------------------------------------------
# Goal contract
# ---------------------------------------------------------------------------


class TestGoalContract:
    def test_is_achieved_within_threshold(self):
        goal = GoalContract(
            objective="test",
            primary_metric="accuracy",
            target_value=0.942,
            success_threshold=0.005,
        )
        assert goal.is_achieved(0.940) is True
        assert goal.is_achieved(0.942) is True

    def test_is_not_achieved_outside_threshold(self):
        goal = GoalContract(
            objective="test",
            primary_metric="accuracy",
            target_value=0.942,
            success_threshold=0.005,
        )
        assert goal.is_achieved(0.93) is False
        assert goal.is_achieved(0.867) is False

    def test_is_achieved_no_target(self):
        goal = GoalContract(objective="test", primary_metric="accuracy")
        assert goal.is_achieved(0.95) is False  # No target = never "achieved"


# ---------------------------------------------------------------------------
# ResearchMemory helpers
# ---------------------------------------------------------------------------


class TestResearchMemory:
    def _make_memory(self) -> ResearchMemory:
        goal = GoalContract(
            objective="Reproduce CIFAR-10",
            primary_metric="accuracy",
            target_value=0.942,
        )
        return ResearchMemory(goal=goal, run_id="test_run_001")

    def test_get_best_result_empty(self):
        mem = self._make_memory()
        assert mem.get_best_result() is None

    def test_update_best(self):
        mem = self._make_memory()
        exp1 = Experiment(id="e1", hypothesis="baseline")
        exp1.observed_result = ExperimentResult(accuracy=0.867)
        exp2 = Experiment(id="e2", hypothesis="with normalize")
        exp2.observed_result = ExperimentResult(accuracy=0.909)
        mem.experiments = [exp1, exp2]
        mem.update_best("accuracy")
        assert mem.best_experiment_id == "e2"

    def test_update_best_respects_highest(self):
        mem = self._make_memory()
        exp1 = Experiment(id="e1")
        exp1.observed_result = ExperimentResult(accuracy=0.95)
        exp2 = Experiment(id="e2")
        exp2.observed_result = ExperimentResult(accuracy=0.91)
        mem.experiments = [exp1, exp2]
        mem.update_best("accuracy")
        assert mem.best_experiment_id == "e1"

    def test_recent_failures_limit(self):
        from research_repro.memory.models import Failure, FailureType
        mem = self._make_memory()
        for i in range(5):
            mem.failures.append(
                Failure(failure_type=FailureType.TOOL_CRASH, description=f"fail {i}")
            )
        assert len(mem.recent_failures(3)) == 3

    def test_plan_advance(self):
        plan = Plan(steps=["step1", "step2", "step3"], current_step_index=0)
        assert plan.current_step == "step1"
        plan.advance()
        assert plan.current_step == "step2"
        plan.advance()
        assert plan.current_step == "step3"
        result = plan.advance()
        assert result is False  # Already at end


# ---------------------------------------------------------------------------
# Experiment graph
# ---------------------------------------------------------------------------


class TestExperimentGraph:
    def _make_exp(self, id: str, parent: str | None = None, acc: float | None = None) -> Experiment:
        exp = Experiment(id=id, parent_id=parent, hypothesis=f"hyp_{id}")
        if acc is not None:
            exp.observed_result = ExperimentResult(accuracy=acc)
        return exp

    def test_roots(self):
        exp1 = self._make_exp("e1")
        exp2 = self._make_exp("e2", parent="e1")
        graph = ExperimentGraph([exp1, exp2])
        roots = graph.roots()
        assert len(roots) == 1
        assert roots[0].id == "e1"

    def test_children(self):
        exp1 = self._make_exp("e1")
        exp2 = self._make_exp("e2", parent="e1")
        exp3 = self._make_exp("e3", parent="e1")
        graph = ExperimentGraph([exp1, exp2, exp3])
        children = graph.children("e1")
        assert {e.id for e in children} == {"e2", "e3"}

    def test_lineage(self):
        exp1 = self._make_exp("e1")
        exp2 = self._make_exp("e2", parent="e1")
        exp3 = self._make_exp("e3", parent="e2")
        graph = ExperimentGraph([exp1, exp2, exp3])
        lineage = graph.lineage("e3")
        assert [e.id for e in lineage] == ["e1", "e2", "e3"]

    def test_best(self):
        exp1 = self._make_exp("e1", acc=0.867)
        exp2 = self._make_exp("e2", acc=0.909)
        exp3 = self._make_exp("e3", acc=0.941)
        graph = ExperimentGraph([exp1, exp2, exp3])
        best = graph.best("accuracy")
        assert best is not None
        assert best.id == "e3"

    def test_already_tried(self):
        exp1 = self._make_exp("e1")
        exp1.hypothesis = "Apply normalization"
        graph = ExperimentGraph([exp1])
        assert graph.already_tried("Apply normalization") is True
        assert graph.already_tried("Apply cosine LR") is False

    def test_len(self):
        graph = ExperimentGraph([self._make_exp("e1"), self._make_exp("e2")])
        assert len(graph) == 2


# ---------------------------------------------------------------------------
# Memory store round-trip
# ---------------------------------------------------------------------------


class TestMemoryStore:
    def test_save_and_load(self, tmp_path):
        goal = GoalContract(
            objective="Reproduce CIFAR-10",
            primary_metric="accuracy",
            target_value=0.942,
        )
        memory = ResearchMemory(goal=goal, run_id="test_store_001")
        memory.current_phase = "experiment_planning"

        store = MemoryStore(tmp_path)
        store.save(memory)

        loaded = store.load()
        assert loaded.run_id == "test_store_001"
        assert loaded.current_phase == "experiment_planning"
        assert loaded.goal.objective == "Reproduce CIFAR-10"

    def test_load_missing_raises(self, tmp_path):
        store = MemoryStore(tmp_path / "nonexistent")
        with pytest.raises(FileNotFoundError):
            store.load()

    def test_exists(self, tmp_path):
        goal = GoalContract(objective="test", primary_metric="accuracy")
        memory = ResearchMemory(goal=goal, run_id="test")
        store = MemoryStore(tmp_path)
        assert store.exists() is False
        store.save(memory)
        assert store.exists() is True
