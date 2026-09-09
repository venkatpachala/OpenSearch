from research_repro.agent.strategy import NaiveStrategy, SelfCorrectingStrategy
from research_repro.evaluation.goal_checker import GoalChecker
from research_repro.memory.models import (
    Failure, FailureType, GoalContract, MetricCriterion, ResearchMemory,
)
from research_repro.recovery import RecoveryOrchestrator


def test_metric_criterion_uses_fractional_tolerance():
    goal = GoalContract(
        objective="reproduce",
        primary_metric="accuracy",
        criterion=MetricCriterion(metric="accuracy", direction="match", target=.942, tolerance=.02),
    )
    assert goal.is_achieved(.94)
    assert not goal.is_achieved(.50)


def test_minimum_and_maximum_criteria():
    assert GoalContract(objective="maximize", criterion=MetricCriterion(metric="accuracy", direction="maximize", minimum=.95)).is_achieved(.96)
    assert not GoalContract(objective="latency", criterion=MetricCriterion(metric="latency", direction="minimize", maximum=100)).is_achieved(101)


def test_reproduction_requires_alignment_and_independent_verification():
    memory = ResearchMemory(goal=GoalContract(objective="reproduce", target_value=.9, criterion=MetricCriterion(metric="accuracy", target=.9, tolerance=.02)))
    facts = GoalChecker().check(memory, {"data": {"accuracy": .9}}, "run_experiment")
    assert facts.reproduction.verdict.value == "inconclusive"
    facts = GoalChecker().check(memory, {"data": {"accuracy": .9, "methodology_aligned": True, "independently_verified": True}}, "run_experiment")
    assert facts.reproduction.verdict.value == "confirmed"


def test_naive_strategy_does_not_consume_recovery_budget():
    memory = ResearchMemory(goal=GoalContract(objective="x"))
    memory.failures.append(Failure(failure_type=FailureType.TOOL_CRASH, description="boom"))
    class Eval:
        failure_type = FailureType.TOOL_CRASH
    class Result:
        evaluation = Eval()
    decision = NaiveStrategy().handle_result(Result(), memory)
    assert decision.action == "retry"
    assert decision.retry_same_action is True
    assert memory.budget.recoveries_consumed == 0
