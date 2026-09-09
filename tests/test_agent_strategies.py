from research_repro.agent.strategy import NaiveStrategy, SelfCorrectingStrategy
from research_repro.memory.models import Failure, FailureType, GoalContract, ResearchMemory
from research_repro.recovery import RecoveryOrchestrator


def test_self_correcting_changes_plan_version_but_naive_does_not():
    class Eval:
        failure_type = FailureType.TOOL_CRASH
    class Result:
        step_index = 1
        tool_name = "run_experiment"
        tool_args = {"parameters": {"batch_size": 8}}
        evaluation = Eval()
        class Response:
            def model_dump(self, mode="json"):
                return {"success": False, "error": "CUDA out of memory"}
        tool_response = Response()

    self_memory = ResearchMemory(goal=GoalContract(objective="x"))
    self_memory.failures.append(Failure(failure_type=FailureType.TOOL_CRASH, description="oom"))
    before = self_memory.plan_version
    decision = SelfCorrectingStrategy(RecoveryOrchestrator()).handle_result(Result(), self_memory)
    assert decision.action == "retry"
    assert decision.retry_same_action is False
    assert decision.repaired_tool_args is not None
    assert decision.repaired_tool_args["parameters"]["batch_size"] == 4
    assert self_memory.plan_version == before + 1

    naive_memory = ResearchMemory(goal=GoalContract(objective="x"))
    naive_memory.failures.append(Failure(failure_type=FailureType.TOOL_CRASH, description="oom"))
    NaiveStrategy().handle_result(Result(), naive_memory)
    assert naive_memory.plan_version == before

