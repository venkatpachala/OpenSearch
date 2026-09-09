"""Pluggable policies for the shared ResearchRepro runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Literal
from pydantic import BaseModel, Field

from ..memory.models import ResearchMemory
from ..recovery import RecoveryContext, RecoveryOrchestrator, RecoveryOutcome


class FailedAction(BaseModel):
    step_index: int
    tool_name: str
    tool_arguments: dict[str, Any] = Field(default_factory=dict)
    failure_type: str | None = None


class StrategyDecision(BaseModel):
    action: Literal["continue", "recover", "replan", "terminate", "retry"]
    success: bool
    reason: str
    retry_same_action: bool = False
    recovery: RecoveryOutcome | None = None
    failed_action: FailedAction | None = None
    repaired_tool_args: dict[str, Any] | None = None


class AgentStrategy(Protocol):
    def handle_result(self, result: Any, memory: ResearchMemory) -> StrategyDecision: ...


class SelfCorrectingStrategy:
    def __init__(self, orchestrator: RecoveryOrchestrator, checkpointer: Any | None = None, independent_evaluator: Any | None = None, schema_repair: Any | None = None):
        self.orchestrator = orchestrator
        self.checkpointer = checkpointer
        self.independent_evaluator = independent_evaluator
        self.schema_repair = schema_repair

    def handle_result(self, result: Any, memory: ResearchMemory) -> StrategyDecision:
        if result.evaluation.failure_type is None or not memory.failures:
            return StrategyDecision(action="continue", success=True, reason="No failure requires recovery")
        context = RecoveryContext(
            memory=memory,
            failure=memory.failures[-1],
            tool_args=result.tool_args,
            tool_response=result.tool_response.model_dump(mode="json"),
            checkpointer=self.checkpointer,
            independent_evaluator=self.independent_evaluator,
            schema_repair=self.schema_repair,
        )
        subtask = (
            result.evaluation.failure_type.value
            if result.evaluation.failure_type is not None
            else result.tool_name
        )
        outcome = self.orchestrator.recover(context, subtask_id=subtask)
        if outcome.success:
            memory.plan_version += 1
            memory.current_plan.version = memory.plan_version
        if not outcome.success:
            return StrategyDecision(action="terminate", success=False, reason=outcome.action_taken, recovery=outcome)
        repaired = outcome.updates.get("tool_args") if isinstance(outcome.updates, dict) else None
        if outcome.retry:
            return StrategyDecision(
                action="retry",
                success=True,
                reason=outcome.action_taken,
                retry_same_action=False,
                recovery=outcome,
                repaired_tool_args=repaired if isinstance(repaired, dict) else None,
            )
        return StrategyDecision(action="recover", success=True, reason=outcome.action_taken, recovery=outcome)


class NaiveStrategy:
    """Control policy: retry the same action without diagnosis or state update."""
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._failed_action: FailedAction | None = None
        self._retries = 0

    def handle_result(self, result: Any, memory: ResearchMemory) -> StrategyDecision:
        if result.evaluation.failure_type is None:
            self._failed_action = None
            self._retries = 0
            return StrategyDecision(action="continue", success=True, reason="No failure observed")
        tool_name = getattr(result, "tool_name", "unknown")
        tool_args = getattr(result, "tool_args", {})
        if self._failed_action is None or self._failed_action.tool_name != tool_name or self._failed_action.tool_arguments != tool_args:
            self._failed_action = FailedAction(step_index=getattr(result, "step_index", 0), tool_name=tool_name, tool_arguments=tool_args.copy(), failure_type=result.evaluation.failure_type.value if result.evaluation.failure_type else None)
            self._retries = 0
        if self._retries >= self.max_retries:
            return StrategyDecision(action="terminate", success=False, reason="Naive retry budget exhausted", failed_action=self._failed_action)
        self._retries += 1
        return StrategyDecision(action="retry", success=True, reason="Retrying the exact same action without diagnosis or plan change", retry_same_action=True, failed_action=self._failed_action)
