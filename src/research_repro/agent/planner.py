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
from pydantic import BaseModel, ValidationError

from ..memory.models import ResearchMemory
from ..tools.base import ToolRegistry
from .planning_context import (
    assess_proposal,
    best_observed_accuracy,
    best_valid_experiment,
    build_failure_context,
    build_planning_evidence,
    canonicalize_tool_args,
    deterministic_next_parameters,
    ensure_falsifiable_hypothesis,
    experiment_parameters,
    rejected_configuration_rows,
    tested_configuration_rows,
    validity_summary,
)


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
You are selecting the next experiment action for a self-correcting scientific agent.

Evaluator evidence is authoritative. You propose one typed tool call. You never
declare success. Empty parameters {} are the sandbox DEFAULT (hidden_size=128,
hidden_layers=2, solver=adam) — that is a real configuration, not "no choice".

When planning_evidence or failure_context is present:
1. Identify the exact failure_type if any.
2. Identify which goal criteria already pass (satisfied_criteria).
3. Identify which criterion failed (failed_criteria / blocking_criterion).
4. Identify observed vs required values from criterion_status. Do not infer this
   solely from raw numbers; use guidance and required_change.
5. Choose a class of intervention from interventions that attacks the blocking
   criterion. Do not hardcode one exact configuration.
6. Do not repeat a previously tested or rejected configuration (see
   tested_configurations, rejected_configurations, fingerprints) unless there is
   an explicit non-determinism reason to retry.
7. Form one falsifiable hypothesis that cites observed evidence, the expected
   effect, and the constraint that must remain satisfied.
8. Change at least one experiment-affecting parameter (hidden_size, hidden_layers,
   max_iter, lr, batch_size, normalize, pca_components, solver).
9. Predict what should change.
10. Select exactly one typed tool action.

Use the exact tool parameter names. For learning rate the valid key is "lr",
never "learning_rate".

GOAL DRIFT: if the primary metric passes but a constraint fails, preserve the
metric and optimize the violated constraint.

ACCURACY BLOCKING: if latency already passes and accuracy fails, preserve the
latency cap and target accuracy. Do not re-run the same low-complexity config.

REGRESSION: if the current experiment is worse than best_valid_experiment, do
not blindly retry it. Branch from the valid configuration or a new parent.

RESULT INCONSISTENCY: do not immediately rerun the same experiment. Use
independent evaluation evidence first.

PHASE-SPECIFIC GUIDANCE:
- "literature_discovery": Call search_literature ONCE.
- "paper_selection": Call retrieve_paper with one paper_id from discovered_papers.
- "method_extraction": Call extract_methodology with the retrieved paper_id.
- "experiment_planning", "running_experiment", "diagnosing", "recovering",
  "replanning", "self_evaluation": Call run_experiment with a NEW experiment_id
  and parameters that differ from rejected fingerprints.
  NEVER call inspect_methodology, search_literature, or retrieve_paper here.

CRITICAL RULES:
1. Choose EXACTLY one tool from available_tools.
2. Provide valid arguments matching the tool's schema exactly.
3. run_experiment parameters must be a dict of concrete knobs, not {}.
4. experiment_id must be unique and must not reuse failed_experiment_ids.
5. best_valid_experiment is the best constraint-satisfying run, NOT the highest
   accuracy if that accuracy violated a constraint.
6. Use validity_summary as the search-space evidence (Pareto-style tradeoff).
7. Reasoning fields must name the failed criterion, the evidence experiment,
   the hypothesis, and the predicted outcome.

Respond ONLY with valid JSON — no markdown, no preamble:
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

    def plan(self, memory: ResearchMemory, registry: ToolRegistry, rejection_hint: str | None = None) -> PlannerOutput:
        """Call the LLM and return a structured next action."""
        context = self._build_context(memory, registry, rejection_hint=rejection_hint)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.5 if rejection_hint else 0.3,
        )
        content = response.choices[0].message.content or "{}"
        content = self._strip_json_fences(content)
        raw = json.loads(content)
        raw = self._coerce_planner_output(raw)
        try:
            output = PlannerOutput(**raw)
        except ValidationError:
            output = PlannerOutput(
                reasoning=ReasoningTrace(
                    goal_relevance="Advance the goal using a new experiment.",
                    evidence_basis="Planner JSON was malformed; using a conservative run_experiment.",
                    hypothesis="A distinct configuration will be evaluated against the contract.",
                    intended_action="run_experiment",
                ),
                selected_tool="run_experiment",
                tool_arguments=raw if isinstance(raw.get("tool_arguments"), dict) else {"parameters": {}},
                confidence=0.3,
            )
        return self._finalize_output(memory, output)

    @staticmethod
    def _coerce_planner_output(raw: dict[str, Any]) -> dict[str, Any]:
        """Accept tool-arg blobs that omit the reasoning envelope."""
        if not isinstance(raw, dict):
            return {
                "reasoning": {
                    "goal_relevance": "Advance the goal contract.",
                    "evidence_basis": "See failure_context and validity_summary.",
                    "hypothesis": "A new configuration will be tested.",
                    "intended_action": "run_experiment",
                },
                "selected_tool": "run_experiment",
                "tool_arguments": {"parameters": {}},
                "confidence": 0.3,
            }
        if "selected_tool" not in raw and ("experiment_id" in raw or "parameters" in raw):
            raw = {
                "reasoning": {
                    "goal_relevance": "Continue the experiment loop.",
                    "evidence_basis": "Planner returned tool arguments without a reasoning envelope.",
                    "hypothesis": "The proposed experiment configuration should be evaluated.",
                    "intended_action": "run_experiment",
                },
                "selected_tool": "run_experiment",
                "tool_arguments": {k: v for k, v in raw.items() if k not in ("reasoning", "confidence", "selected_tool")},
                "confidence": float(raw.get("confidence", 0.4) or 0.4),
            }
        raw.setdefault("selected_tool", "run_experiment")
        raw.setdefault("tool_arguments", {})
        raw.setdefault("confidence", 0.5)
        if "reasoning" not in raw or not isinstance(raw.get("reasoning"), dict):
            raw["reasoning"] = {
                "goal_relevance": "Advance the goal contract.",
                "evidence_basis": "See failure_context and validity_summary.",
                "hypothesis": "A new configuration will be tested.",
                "intended_action": str(raw.get("selected_tool") or "run_experiment"),
            }
        return raw

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

    def _build_context(self, memory: ResearchMemory, registry: ToolRegistry, rejection_hint: str | None = None) -> str:
        """Serialize the relevant memory state for the LLM."""
        recent_experiments = [
            {
                "id": exp.id,
                "parent_id": exp.parent_id,
                "hypothesis": exp.hypothesis,
                "parameters": exp.parameters,
                "expected_result": exp.expected_result,
                "config_fingerprint": exp.config_fingerprint,
                "execution_status": exp.execution_status.value,
                "evaluation_status": exp.evaluation_status.value,
                "result": (
                    exp.observed_result.model_dump() if exp.observed_result else None
                ),
                "failure_id": exp.failure_id,
            }
            for exp in memory.recent_experiments(5)
        ]

        recent_failures = [
            {
                "failure_type": f.failure_type.value,
                "description": f.description,
                "experiment_id": f.experiment_id,
                "recovered": f.recovered,
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
                "criterion": memory.goal.criterion.model_dump() if memory.goal.criterion else None,
                "constraints": [c.model_dump() for c in memory.goal.constraints],
            },
            "current_phase": memory.current_phase,
            "plan_version": memory.plan_version,
            "current_plan": {
                "steps": memory.current_plan.steps,
                "current_step_index": memory.current_plan.current_step_index,
                "current_step": memory.current_plan.current_step,
            },
            "discovered_papers": memory.recent_discoveries(5),
            "selected_paper": (
                memory.selected_paper.model_dump() if memory.selected_paper else None
            ),
            "methodology": (
                memory.methodology.model_dump() if memory.methodology else None
            ),
            "recent_experiments": recent_experiments,
            "recent_failures": recent_failures,
            "failed_experiment_ids": [
                f.experiment_id for f in memory.failures if f.experiment_id
            ],
            "best_experiment_id": memory.best_experiment_id,
            "best_valid_experiment": best_valid_experiment(memory),
            "best_observed_accuracy": best_observed_accuracy(memory),
            "planning_evidence": build_planning_evidence(memory).model_dump(mode="json"),
            "failure_context": (
                ctx.model_dump(mode="json") if (ctx := build_failure_context(memory)) else None
            ),
            "tested_configurations": tested_configuration_rows(memory),
            "rejected_configurations": rejected_configuration_rows(memory),
            "validity_summary": validity_summary(memory),
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
            "reproduction_assessment": memory.reproduction.model_dump(),
            "unresolved_questions": memory.unresolved_questions,
            "budget": {
                "experiments_consumed": memory.budget.experiments_consumed,
                "max_experiments": memory.goal.max_experiments,
                "recoveries_consumed": memory.budget.recoveries_consumed,
                "max_recoveries": memory.goal.allowed_resources.max_total_recoveries,
            },
            "available_tools": registry.list_available(),
        }
        if rejection_hint:
            evidence = context["planning_evidence"]
            context["proposal_rejection"] = {
                "blocked": True,
                "reason": rejection_hint,
                "instruction": (
                    "Your previous proposal was rejected and was NOT executed. "
                    "Do not repeat a tested or rejected fingerprint. "
                    f"required_change={evidence.get('required_change')}. "
                    f"blocking_criterion={evidence.get('blocking_criterion')}. "
                    f"guidance={evidence.get('guidance')}. "
                    "Change at least one of hidden_size, hidden_layers, pca_components, "
                    "normalize, max_iter, or lr. batch_size does not reduce parameter-count latency. "
                    "Empty parameters {} is the already-tested default 128x2 run."
                ),
            }
        return json.dumps(context, indent=2, default=str)

    def _finalize_output(self, memory: ResearchMemory, output: PlannerOutput) -> PlannerOutput:
        args = canonicalize_tool_args(output.tool_arguments if isinstance(output.tool_arguments, dict) else {})
        params = experiment_parameters(args)
        hypothesis = ensure_falsifiable_hypothesis(memory, output.reasoning.hypothesis, params)
        evidence = build_planning_evidence(memory)
        if not output.reasoning.evidence_basis or "see failure_context" in output.reasoning.evidence_basis.lower():
            evidence_basis = evidence.guidance or output.reasoning.evidence_basis
        else:
            evidence_basis = output.reasoning.evidence_basis
        output.tool_arguments = args
        output.reasoning.hypothesis = hypothesis
        output.reasoning.evidence_basis = evidence_basis
        if output.selected_tool == "run_experiment":
            blocked = assess_proposal(memory, output.selected_tool, args)
            if blocked.blocked:
                fallback = deterministic_next_parameters(memory)
                if fallback:
                    args["parameters"] = fallback
                    if not args.get("experiment_id"):
                        args["experiment_id"] = f"exp_{memory.budget.experiments_consumed + 1:02d}"
                    output.tool_arguments = args
                    output.reasoning.hypothesis = ensure_falsifiable_hypothesis(memory, "", fallback)
                    output.reasoning.intended_action = f"run_experiment {fallback}"
                    output.reasoning.evidence_basis = (
                        evidence.guidance
                        + " Previous proposal was blocked; selected a different intervention class from evidence."
                    )
        return output
