"""
Planner — the LLM-based hypothesis and action engine.

The planner receives the full research state and produces a structured
PlannerOutput: which tool to call, with what arguments, and a typed
ReasoningTrace explaining the decision.

The reasoning trace is NOT raw chain-of-thought — it is a structured
decision rationale with four fields that map directly to observable state:
  goal_relevance   — why this action advances the goal
  evidence_basis   — what evidence supports this choice
  hypothesis       — what we expect to learn/achieve
  intended_action  — plain-language description

This trace is persisted to JSONL and shown in the run viewer.
"""
from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from ..memory.models import ResearchMemory
from ..tools.base import ToolRegistry


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class ReasoningTrace(BaseModel):
    """Structured decision rationale — the visible 'thinking' of the agent."""

    goal_relevance: str
    """Why does this action advance the stated goal?"""

    evidence_basis: str
    """What evidence from memory or prior results supports this choice?"""

    hypothesis: str
    """What do we expect to learn or achieve from this action?"""

    intended_action: str
    """Plain-language description of what we're about to do."""


class PlannerOutput(BaseModel):
    reasoning: ReasoningTrace
    selected_tool: str
    tool_arguments: dict[str, Any]
    confidence: float


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the planning engine of a self-correcting scientific experiment agent.

Your role: given the current research state, decide the SINGLE BEST next action
to advance the goal toward reproducing and verifying a scientific result.

Rules:
1. Choose EXACTLY one tool from the available_tools list.
2. Provide valid arguments that match the tool's parameter schema exactly.
3. Provide a structured reasoning trace (goal_relevance, evidence_basis,
   hypothesis, intended_action).
4. Never choose an action identical to one that already failed with no new information.
5. If a discrepancy exists, investigate its cause before re-running the experiment.
6. If budget is nearly exhausted, prioritize actions most likely to achieve the goal.

You are NOT responsible for evaluating results — that is the evaluator's job.
Focus on the best possible next action given current evidence.

Respond ONLY with valid JSON — no markdown, no preamble. Match this exact structure:
{
  "reasoning": {
    "goal_relevance": "...",
    "evidence_basis": "...",
    "hypothesis": "...",
    "intended_action": "..."
  },
  "selected_tool": "<tool_name>",
  "tool_arguments": { ... },
  "confidence": 0.0
}
"""


class Planner:
    """
    LLM-based planner that produces structured PlannerOutput.

    The planner sees:
      - Goal contract
      - Current phase and plan
      - Recent experiments (last 3)
      - Recent failures
      - Available tools with schemas
      - Budget state

    It produces a typed next action.
    """

    def __init__(
        self,
        api_key: str = "ollama",
        model: str = "qwen2.5-coder:7b",
        base_url: str | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key or "ollama", base_url=base_url)
        self.model = model

    def plan(self, memory: ResearchMemory, registry: ToolRegistry) -> PlannerOutput:
        """Call the LLM and return a structured next action."""
        context = self._build_context(memory, registry)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
        content = response.choices[0].message.content or "{}"
        content = self._strip_json_fences(content)
        raw = json.loads(content)
        return PlannerOutput(**raw)

    @staticmethod
    def _strip_json_fences(text: str) -> str:
        s = text.strip()
        if s.startswith("```json"):
            s = s[7:]
        elif s.startswith("```"):
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
        return s.strip()

    def _build_context(self, memory: ResearchMemory, registry: ToolRegistry) -> str:
        """Serialize the relevant memory state for the LLM."""
        recent_experiments = [
            {
                "id": exp.id,
                "hypothesis": exp.hypothesis,
                "execution_status": exp.execution_status.value,
                "evaluation_status": exp.evaluation_status.value,
                "result": (
                    exp.observed_result.model_dump() if exp.observed_result else None
                ),
                "failure_id": exp.failure_id,
            }
            for exp in memory.recent_experiments(3)
        ]

        recent_failures = [
            {
                "failure_type": f.failure_type.value,
                "description": f.description,
                "experiment_id": f.experiment_id,
            }
            for f in memory.recent_failures(3)
        ]

        recent_evidence = [
            {
                "claim": e.claim,
                "source_type": e.source_type,
                "location": e.location,
            }
            for e in memory.evidence[-5:]
        ]

        context: dict[str, Any] = {
            "goal": {
                "objective": memory.goal.objective,
                "primary_metric": memory.goal.primary_metric,
                "target_value": memory.goal.target_value,
                "success_threshold": memory.goal.success_threshold,
                "constraints": [c.model_dump() for c in memory.goal.constraints],
            },
            "current_phase": memory.current_phase,
            "plan_version": memory.plan_version,
            "current_plan": {
                "steps": memory.current_plan.steps,
                "current_step_index": memory.current_plan.current_step_index,
                "current_step": memory.current_plan.current_step,
            },
            "selected_paper": (
                memory.selected_paper.model_dump() if memory.selected_paper else None
            ),
            "methodology": (
                memory.methodology.model_dump() if memory.methodology else None
            ),
            "recent_experiments": recent_experiments,
            "recent_failures": recent_failures,
            "recent_evidence": recent_evidence,
            "discrepancies": [
                {
                    "metric": d.metric,
                    "expected": d.expected,
                    "observed": d.observed,
                    "magnitude": d.magnitude,
                    "possible_causes": d.possible_causes,
                }
                for d in memory.discrepancies[-3:]
            ],
            "unresolved_questions": memory.unresolved_questions,
            "budget": {
                "experiments_consumed": memory.budget.experiments_consumed,
                "max_experiments": memory.goal.max_experiments,
                "recoveries_consumed": memory.budget.recoveries_consumed,
                "max_recoveries": memory.goal.allowed_resources.max_total_recoveries,
            },
            "available_tools": registry.list_available(),
        }
        return json.dumps(context, indent=2, default=str)
