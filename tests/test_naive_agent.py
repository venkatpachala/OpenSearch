from research_repro.agent.strategy import NaiveStrategy
from research_repro.memory.models import Failure, FailureType, GoalContract, ResearchMemory


class FakeEvaluation:
    failure_type = FailureType.TOOL_CRASH


class FakeResult:
    step_index = 3
    tool_name = "run_experiment"
    tool_args = {"experiment_id": "exp_1", "parameters": {"batch_size": 8}}
    evaluation = FakeEvaluation()


def test_naive_stores_and_reuses_exact_failed_action():
    memory = ResearchMemory(goal=GoalContract(objective="x"))
    memory.failures.append(Failure(failure_type=FailureType.TOOL_CRASH, description="failure"))
    decision = NaiveStrategy(max_retries=1).handle_result(FakeResult(), memory)
    assert decision.retry_same_action
    assert decision.failed_action.tool_name == "run_experiment"
    assert decision.failed_action.tool_arguments == FakeResult.tool_args

