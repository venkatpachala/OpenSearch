"""
Enhanced Self-Evaluator — two-stage evaluation pipeline.

Stage 1 (Deterministic):
  - GoalChecker: metric achievement, constraint satisfaction, regression, budget
  - DiscrepancyAnalyzer: gap magnitude and causal analysis
  - FailureTaxonomyClassifier: rule-based failure type classification

Stage 2 (LLM Synthesis):
  - Receives all Stage 1 facts as structured context
  - Synthesizes a final decision and rationale
  - Cannot contradict hard facts (e.g., can't say "no failure" if exit code != 0)

This architecture means:
  - The LLM cannot hallucinate whether a constraint was violated (GoalChecker knows)
  - The LLM cannot miss an obvious failure type (FailureTaxonomy pre-classifies)
  - The LLM adds value on ambiguous cases and rationale generation
  - The system degrades gracefully if the LLM fails (falls back to Stage 1 results)
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from ..memory.models import FailureType, ResearchMemory
from .goal_checker import GoalChecker, GoalCheckResult
from .discrepancy import DiscrepancyAnalyzer
from .failure_taxonomy import FailureTaxonomyClassifier, TaxonomyResult


# ---------------------------------------------------------------------------
# Evaluation enumerations (unchanged from Phase 2)
# ---------------------------------------------------------------------------


class GoalProgress(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class ConstraintStatus(str, Enum):
    ALL_MET = "all_met"
    VIOLATED = "violated"
    AT_RISK = "at_risk"


class HypothesisStatus(str, Enum):
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class Decision(str, Enum):
    CONTINUE = "continue"
    DIAGNOSE_AND_RECOVER = "diagnose_and_recover"
    GOAL_ACHIEVED = "goal_achieved"
    TERMINATE = "terminate"


# ---------------------------------------------------------------------------
# Enhanced EvaluationResult
# ---------------------------------------------------------------------------


class EvaluationResult(BaseModel):
    execution_success: bool
    output_valid: bool
    metric_valid: bool
    goal_progress: GoalProgress
    constraint_status: ConstraintStatus
    hypothesis_status: HypothesisStatus
    failure_type: FailureType | None = None
    decision: Decision
    confidence: float
    rationale: str

    # Phase 5 additions — deterministic pre-check results
    goal_check: dict | None = None
    """Serialized GoalCheckResult — facts used to make this evaluation."""

    taxonomy_result: dict | None = None
    """Serialized TaxonomyResult — rule-based classification."""

    discrepancy_analysis: dict | None = None
    """Serialized DiscrepancyAnalysis — if a discrepancy was detected."""


# ---------------------------------------------------------------------------
# System prompt (enhanced with Stage 1 context)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the independent evaluation engine of a self-correcting scientific experiment agent.

Your role: synthesize the pre-computed deterministic facts below into a final evaluation.
You MUST NOT contradict the pre-computed facts. Use them as ground truth.

Pre-computed facts will be provided in the user message under "DETERMINISTIC_FACTS".
These facts are computed from rules — they are authoritative.

Your job:
1. If deterministic facts indicate a specific failure_type, use that type (don't invent another).
2. If deterministic facts show metric_achieved=true AND no constraints violated AND no regression,
   set decision=goal_achieved.
3. If budget_exhausted=true, set decision=terminate.
4. Otherwise synthesize goal_progress, hypothesis_status, and a clear rationale.

Be concise. Be honest. Don't rationalize failures.

Respond ONLY with valid JSON:
{
  "execution_success": true/false,
  "output_valid": true/false,
  "metric_valid": true/false,
  "goal_progress": "positive|negative|neutral|unknown",
  "constraint_status": "all_met|violated|at_risk",
  "hypothesis_status": "supported|rejected|inconclusive",
  "failure_type": null or "tool_crash|tool_schema_error|result_inconsistency|goal_drift|regression|non_determinism",
  "decision": "continue|diagnose_and_recover|goal_achieved|terminate",
  "confidence": 0.0-1.0,
  "rationale": "One clear sentence"
}
"""


class SelfEvaluator:
    """
    Two-stage self-evaluator.

    Stage 1: GoalChecker + DiscrepancyAnalyzer + FailureTaxonomyClassifier
    Stage 2: LLM synthesis with Stage 1 facts as input context.

    Falls back to Stage 1 only if the LLM call fails.
    """

    def __init__(
        self,
        api_key: str = "ollama",
        model: str = "qwen2.5-coder:7b",
        base_url: str | None = None,
    ) -> None:
        self.client = OpenAI(api_key=api_key or "ollama", base_url=base_url)
        self.model = model
        self.goal_checker = GoalChecker()
        self.discrepancy_analyzer = DiscrepancyAnalyzer()
        self.taxonomy_classifier = FailureTaxonomyClassifier()

    def evaluate(
        self,
        memory: ResearchMemory,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_response: dict[str, Any],
        hypothesis: str,
    ) -> EvaluationResult:
        """
        Two-stage evaluation of a tool call result.

        Stage 1 is always run (deterministic, fast, no cost).
        Stage 2 (LLM) is run with Stage 1 facts pre-loaded.
        """
        # ── Stage 1: Deterministic checks ────────────────────────────────────
        goal_check = self.goal_checker.check(memory, tool_response, tool_name)
        goal_facts = goal_check.to_dict()

        taxonomy = self.taxonomy_classifier.classify(
            tool_name=tool_name,
            tool_response=tool_response,
            goal_check_facts=goal_facts,
        )

        # Check for discrepancy (only for compare_results tool)
        discrepancy_analysis = None
        if tool_name == "compare_results":
            data = tool_response.get("data", tool_response)
            if data.get("status") == "DISCREPANCY":
                discrepancy_analysis = self.discrepancy_analyzer.analyze(
                    metric=memory.goal.primary_metric,
                    expected=data.get("reported", 0),
                    observed=data.get("observed", 0),
                    methodology=memory.methodology,
                )

        # ── Shortcut: hard termination conditions (no LLM needed) ─────────
        if goal_facts.get("budget_exhausted"):
            return self._make_terminate_result(
                goal_check, taxonomy, discrepancy_analysis,
                "Budget exhausted — max experiments reached"
            )

        if goal_check.all_good:
            return self._make_success_result(goal_check, taxonomy, discrepancy_analysis)

        # ── Stage 2: LLM synthesis ────────────────────────────────────────────
        try:
            return self._llm_evaluate(
                memory, tool_name, tool_args, tool_response,
                hypothesis, goal_facts, taxonomy,
                discrepancy_analysis.to_dict() if discrepancy_analysis else None,
            )
        except Exception:
            # Graceful fallback to Stage 1 if LLM fails
            return self._fallback_result(goal_check, taxonomy, discrepancy_analysis)

    # ── Stage 2: LLM call ────────────────────────────────────────────────────

    def _llm_evaluate(
        self,
        memory: ResearchMemory,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_response: dict[str, Any],
        hypothesis: str,
        goal_facts: dict,
        taxonomy: TaxonomyResult,
        discrepancy: dict | None,
    ) -> EvaluationResult:
        context = self._build_context(
            memory, tool_name, tool_args, tool_response,
            hypothesis, goal_facts, taxonomy, discrepancy,
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": context},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content or "{}"
        content = self._strip_json_fences(content)
        raw = json.loads(content)

        # Enforce: if taxonomy pre-classified a failure type, trust it
        if taxonomy.is_failure and raw.get("failure_type") is None:
            raw["failure_type"] = taxonomy.failure_type.value
            raw["confidence"] = min(raw.get("confidence", 0.7), taxonomy.confidence)

        return EvaluationResult(
            **raw,
            goal_check=goal_facts,
            taxonomy_result={
                "failure_type": taxonomy.failure_type.value if taxonomy.failure_type else None,
                "confidence": taxonomy.confidence,
                "reason": taxonomy.reason,
            },
            discrepancy_analysis=discrepancy,
        )

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

    def _build_context(
        self,
        memory: ResearchMemory,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_response: dict[str, Any],
        hypothesis: str,
        goal_facts: dict,
        taxonomy: TaxonomyResult,
        discrepancy: dict | None,
    ) -> str:
        context: dict[str, Any] = {
            "DETERMINISTIC_FACTS": {
                "goal_facts": goal_facts,
                "taxonomy": {
                    "pre_classified_failure": (
                        taxonomy.failure_type.value if taxonomy.failure_type else None
                    ),
                    "taxonomy_confidence": taxonomy.confidence,
                    "taxonomy_reason": taxonomy.reason,
                },
                "discrepancy": discrepancy,
            },
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_response": tool_response,
            "hypothesis": hypothesis,
            "current_phase": memory.current_phase,
            "total_failures": len(memory.failures),
            "total_experiments": len(memory.experiments),
        }
        return json.dumps(context, indent=2, default=str)

    # ── Shortcut result constructors ─────────────────────────────────────────

    def _make_success_result(
        self, goal_check: GoalCheckResult, taxonomy: TaxonomyResult, disc=None
    ) -> EvaluationResult:
        return EvaluationResult(
            execution_success=True,
            output_valid=True,
            metric_valid=True,
            goal_progress=GoalProgress.POSITIVE,
            constraint_status=ConstraintStatus.ALL_MET,
            hypothesis_status=HypothesisStatus.SUPPORTED,
            failure_type=None,
            decision=Decision.GOAL_ACHIEVED,
            confidence=0.97,
            rationale=(
                f"Goal achieved: {goal_check.metric.metric}="
                f"{goal_check.metric.observed_value} is within "
                f"{goal_check.metric.success_threshold} of target "
                f"{goal_check.metric.target_value}"
            ),
            goal_check=goal_check.to_dict(),
            taxonomy_result=None,
            discrepancy_analysis=disc.to_dict() if disc else None,
        )

    def _make_terminate_result(
        self, goal_check: GoalCheckResult, taxonomy: TaxonomyResult,
        disc=None, reason: str = ""
    ) -> EvaluationResult:
        return EvaluationResult(
            execution_success=True,
            output_valid=True,
            metric_valid=goal_check.metric.observed_value is not None,
            goal_progress=GoalProgress.NEUTRAL,
            constraint_status=ConstraintStatus.ALL_MET,
            hypothesis_status=HypothesisStatus.INCONCLUSIVE,
            failure_type=None,
            decision=Decision.TERMINATE,
            confidence=0.99,
            rationale=reason or "Budget exhausted",
            goal_check=goal_check.to_dict(),
            taxonomy_result=None,
            discrepancy_analysis=None,
        )

    def _fallback_result(
        self, goal_check: GoalCheckResult, taxonomy: TaxonomyResult, disc=None
    ) -> EvaluationResult:
        """Stage 1 fallback when LLM call fails."""
        decision = Decision.CONTINUE
        if taxonomy.is_failure:
            decision = Decision.DIAGNOSE_AND_RECOVER
        if goal_check.all_good:
            decision = Decision.GOAL_ACHIEVED

        return EvaluationResult(
            execution_success=not taxonomy.is_failure
            or taxonomy.failure_type == FailureType.RESULT_INCONSISTENCY,
            output_valid=True,
            metric_valid=goal_check.metric.observed_value is not None,
            goal_progress=(
                GoalProgress.POSITIVE if goal_check.metric.achieved
                else GoalProgress.NEGATIVE if taxonomy.is_failure
                else GoalProgress.NEUTRAL
            ),
            constraint_status=(
                ConstraintStatus.VIOLATED
                if goal_check.any_constraint_violated
                else ConstraintStatus.ALL_MET
            ),
            hypothesis_status=(
                HypothesisStatus.REJECTED
                if taxonomy.is_failure
                else HypothesisStatus.INCONCLUSIVE
            ),
            failure_type=taxonomy.failure_type,
            decision=decision,
            confidence=0.70,
            rationale=(
                f"[Fallback — LLM unavailable] "
                + (taxonomy.reason if taxonomy.is_failure else "No failure detected")
            ),
            goal_check=goal_check.to_dict(),
            taxonomy_result={
                "failure_type": (
                    taxonomy.failure_type.value if taxonomy.failure_type else None
                ),
                "confidence": taxonomy.confidence,
                "reason": taxonomy.reason,
            },
            discrepancy_analysis=disc.to_dict() if disc else None,
        )
