"""
ReAct Controller — orchestrates one complete REASON → ACT → OBSERVE → EVALUATE cycle.

Every step follows this exact sequence:
  1. REASON  — Planner produces a typed PlannerOutput (which tool, which args, why)
  2. ACT     — ToolRegistry validates args and executes the tool
  3. OBSERVE — Structure the tool response (or schema error feedback)
  4. EVALUATE — SelfEvaluator independently judges the result

The output (StepResult) is fed to the AgentLoop, which decides what to do next.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from .planner import Planner, PlannerOutput, ReasoningTrace
from ..evaluation.evaluator import Decision, EvaluationResult, SelfEvaluator
from ..memory.models import ResearchMemory
from ..observability.events import EventType
from ..observability.logger import EventLogger
from ..tools.base import ToolRegistry, ToolResponse, ToolSchemaError


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------


class StepResult(BaseModel):
    """The complete record of one REASON → ACT → OBSERVE → EVALUATE cycle."""

    step_index: int
    reasoning: ReasoningTrace
    tool_name: str
    tool_args: dict[str, Any]
    tool_response: ToolResponse
    evaluation: EvaluationResult
    decision: Decision

    @property
    def was_schema_error(self) -> bool:
        return self.tool_response.error == "TOOL_SCHEMA_ERROR"

    @property
    def succeeded(self) -> bool:
        return self.tool_response.success and self.evaluation.execution_success


# ---------------------------------------------------------------------------
# ReAct Controller
# ---------------------------------------------------------------------------


class ReactController:
    """
    Orchestrates one ReAct step.

    Dependencies are injected — the controller doesn't know about
    LLM keys, file paths, or run IDs. It only knows about the four
    components of the ReAct loop.
    """

    def __init__(
        self,
        planner: Planner,
        evaluator: SelfEvaluator,
        registry: ToolRegistry,
        logger: EventLogger,
    ) -> None:
        self.planner = planner
        self.evaluator = evaluator
        self.registry = registry
        self.logger = logger

    def step(self, memory: ResearchMemory) -> StepResult:
        """Execute one full REASON → ACT → OBSERVE → EVALUATE cycle."""
        step_index = memory.step_count + 1

        # ── 1. REASON ────────────────────────────────────────────────────────
        planner_output = self._reason(memory, step_index)

        # ── 2. ACT ───────────────────────────────────────────────────────────
        tool_response, schema_error = self._act(planner_output, step_index)

        # ── 3. OBSERVE ───────────────────────────────────────────────────────
        # Schema errors produce structured feedback instead of a real response
        if schema_error is not None:
            tool_response = ToolResponse(
                success=False,
                error="TOOL_SCHEMA_ERROR",
                data=schema_error.to_feedback(),
            )

        self._observe(planner_output, tool_response, step_index)

        # ── 4. EVALUATE ──────────────────────────────────────────────────────
        evaluation = self._evaluate(memory, planner_output, tool_response, step_index)

        return StepResult(
            step_index=step_index,
            reasoning=planner_output.reasoning,
            tool_name=planner_output.selected_tool,
            tool_args=planner_output.tool_arguments,
            tool_response=tool_response,
            evaluation=evaluation,
            decision=evaluation.decision,
        )

    # ── Phase implementations ────────────────────────────────────────────────

    def _reason(self, memory: ResearchMemory, step_index: int) -> PlannerOutput:
        planner_output = self.planner.plan(memory, self.registry)
        self.logger.log(
            EventType.REASON,
            step=step_index,
            goal_relevance=planner_output.reasoning.goal_relevance,
            evidence_basis=planner_output.reasoning.evidence_basis,
            hypothesis=planner_output.reasoning.hypothesis,
            intended_action=planner_output.reasoning.intended_action,
            confidence=planner_output.confidence,
        )
        self.logger.log(
            EventType.ACTION,
            step=step_index,
            selected_tool=planner_output.selected_tool,
            tool_arguments=planner_output.tool_arguments,
        )
        return planner_output

    def _act(
        self, planner_output: PlannerOutput, step_index: int
    ) -> tuple[ToolResponse, ToolSchemaError | None]:
        tool_name = planner_output.selected_tool
        tool_args = planner_output.tool_arguments

        self.logger.log(
            EventType.TOOL_CALL,
            step=step_index,
            tool_name=tool_name,
            args=tool_args,
        )

        schema_error: ToolSchemaError | None = None
        try:
            response = self.registry.execute(tool_name, tool_args)
        except ToolSchemaError as e:
            schema_error = e
            response = ToolResponse(success=False, error=str(e))
            self.logger.log(
                EventType.TOOL_SCHEMA_ERROR,
                step=step_index,
                tool_name=tool_name,
                field_errors=e.field_errors,
                feedback=e.to_feedback(),
            )
        except KeyError as e:
            response = ToolResponse.fail(f"Unknown tool: {e}")

        self.logger.log(
            EventType.TOOL_RESULT,
            step=step_index,
            tool_name=tool_name,
            success=response.success,
            error=response.error,
            data_keys=list(response.data.keys()),
        )
        return response, schema_error

    def _observe(
        self,
        planner_output: PlannerOutput,
        tool_response: ToolResponse,
        step_index: int,
    ) -> None:
        self.logger.log(
            EventType.OBSERVATION,
            step=step_index,
            tool_name=planner_output.selected_tool,
            success=tool_response.success,
            summary=(
                f"Tool '{planner_output.selected_tool}' "
                + ("succeeded" if tool_response.success else f"failed: {tool_response.error}")
            ),
        )

    def _evaluate(
        self,
        memory: ResearchMemory,
        planner_output: PlannerOutput,
        tool_response: ToolResponse,
        step_index: int,
    ) -> EvaluationResult:
        evaluation = self.evaluator.evaluate(
            memory=memory,
            tool_name=planner_output.selected_tool,
            tool_args=planner_output.tool_arguments,
            tool_response=tool_response.model_dump(),
            hypothesis=planner_output.reasoning.hypothesis,
        )
        self.logger.log(
            EventType.SELF_EVALUATION,
            step=step_index,
            execution_success=evaluation.execution_success,
            output_valid=evaluation.output_valid,
            metric_valid=evaluation.metric_valid,
            goal_progress=evaluation.goal_progress.value,
            constraint_status=evaluation.constraint_status.value,
            hypothesis_status=evaluation.hypothesis_status.value,
            failure_type=(
                evaluation.failure_type.value if evaluation.failure_type else None
            ),
            decision=evaluation.decision.value,
            confidence=evaluation.confidence,
            rationale=evaluation.rationale,
        )
        return evaluation
