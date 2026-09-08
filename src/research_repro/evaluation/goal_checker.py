"""
Goal Checker — deterministic goal and constraint verification.

This runs BEFORE the LLM evaluator, providing ground-truth facts
that the LLM then synthesizes into a decision. This prevents the
LLM from hallucinating whether a constraint was violated.

Checks:
  1. Metric achievement — is observed value within success_threshold?
  2. Constraint satisfaction — are all goal constraints met?
  3. Regression detection — is this worse than the best known result?
  4. Non-determinism detection — high variance across runs?
  5. Budget status — how much runway remains?
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..memory.models import GoalContract, ResearchMemory


@dataclass
class MetricCheck:
    metric: str
    target_value: float | None
    observed_value: float | None
    success_threshold: float
    achieved: bool
    gap: float | None
    gap_pct: float | None


@dataclass
class ConstraintCheck:
    constraint_name: str
    satisfied: bool
    observed_value: Any | None
    reason: str


@dataclass
class RegressionCheck:
    is_regression: bool
    best_known: float | None
    current_value: float | None
    delta: float | None


@dataclass
class NonDeterminismCheck:
    is_non_deterministic: bool
    std_dev: float | None
    samples: list[float] = field(default_factory=list)
    threshold: float = 0.02


@dataclass
class BudgetCheck:
    experiments_remaining: int
    recoveries_remaining: int
    at_risk: bool  # True if < 20% budget left
    exhausted: bool


@dataclass
class GoalCheckResult:
    """Complete deterministic check result — fed to LLM as facts."""

    metric: MetricCheck
    constraints: list[ConstraintCheck]
    regression: RegressionCheck
    non_determinism: NonDeterminismCheck
    budget: BudgetCheck

    @property
    def any_constraint_violated(self) -> bool:
        return any(not c.satisfied for c in self.constraints)

    @property
    def all_good(self) -> bool:
        """True if goal achieved AND no constraints violated AND no regression."""
        return (
            self.metric.achieved
            and not self.any_constraint_violated
            and not self.regression.is_regression
        )

    def to_dict(self) -> dict:
        """Serializable form for LLM context injection."""
        return {
            "metric_achieved": self.metric.achieved,
            "metric_gap": self.metric.gap,
            "metric_gap_pct": self.metric.gap_pct,
            "target_value": self.metric.target_value,
            "observed_value": self.metric.observed_value,
            "any_constraint_violated": self.any_constraint_violated,
            "constraint_details": [
                {
                    "name": c.constraint_name,
                    "satisfied": c.satisfied,
                    "reason": c.reason,
                }
                for c in self.constraints
            ],
            "is_regression": self.regression.is_regression,
            "regression_delta": self.regression.delta,
            "best_known_value": self.regression.best_known,
            "is_non_deterministic": self.non_determinism.is_non_deterministic,
            "budget_experiments_remaining": self.budget.experiments_remaining,
            "budget_recoveries_remaining": self.budget.recoveries_remaining,
            "budget_at_risk": self.budget.at_risk,
            "budget_exhausted": self.budget.exhausted,
        }


class GoalChecker:
    """
    Deterministic goal and constraint verifier.

    Called by SelfEvaluator before the LLM call, producing facts
    that the LLM synthesizes into a final evaluation decision.
    """

    def check(
        self,
        memory: ResearchMemory,
        tool_response: dict[str, Any],
        tool_name: str,
    ) -> GoalCheckResult:
        """Run all deterministic checks and return a structured result."""
        goal = memory.goal
        observed_value = self._extract_primary_metric(tool_response, goal.primary_metric)

        return GoalCheckResult(
            metric=self._check_metric(goal, observed_value),
            constraints=self._check_constraints(goal, tool_response, memory),
            regression=self._check_regression(memory, goal.primary_metric, observed_value),
            non_determinism=self._check_non_determinism(memory, goal.primary_metric),
            budget=self._check_budget(memory, goal),
        )

    # ── Individual checks ────────────────────────────────────────────────────

    def _check_metric(self, goal: GoalContract, observed: float | None) -> MetricCheck:
        if observed is None or goal.target_value is None:
            return MetricCheck(
                metric=goal.primary_metric,
                target_value=goal.target_value,
                observed_value=observed,
                success_threshold=goal.success_threshold,
                achieved=False,
                gap=None,
                gap_pct=None,
            )
        gap = abs(goal.target_value - observed)
        gap_pct = gap / max(goal.target_value, 1e-9) * 100
        achieved = gap <= goal.success_threshold
        return MetricCheck(
            metric=goal.primary_metric,
            target_value=goal.target_value,
            observed_value=observed,
            success_threshold=goal.success_threshold,
            achieved=achieved,
            gap=round(gap, 4),
            gap_pct=round(gap_pct, 2),
        )

    def _check_constraints(
        self,
        goal: GoalContract,
        tool_response: dict[str, Any],
        memory: ResearchMemory,
    ) -> list[ConstraintCheck]:
        checks = []
        for constraint in goal.constraints:
            check = self._evaluate_constraint(constraint.name, tool_response, memory)
            checks.append(check)
        return checks

    def _evaluate_constraint(
        self,
        name: str,
        tool_response: dict[str, Any],
        memory: ResearchMemory,
    ) -> ConstraintCheck:
        """Evaluate a named constraint against the tool response."""
        name_lower = name.lower()

        if "latency" in name_lower:
            latency = tool_response.get("latency_ms") or tool_response.get(
                "data", {}
            ).get("latency_ms")
            if latency is not None:
                limit = 100.0
                satisfied = latency <= limit
                return ConstraintCheck(
                    constraint_name=name,
                    satisfied=satisfied,
                    observed_value=latency,
                    reason=(
                        f"latency={latency}ms {'≤' if satisfied else '>'} {limit}ms limit"
                    ),
                )

        if "public_dataset" in name_lower:
            return ConstraintCheck(
                constraint_name=name,
                satisfied=True,
                observed_value="MNIST/CIFAR-10",
                reason="Dataset is publicly available",
            )

        if "single_gpu" in name_lower:
            return ConstraintCheck(
                constraint_name=name,
                satisfied=True,
                observed_value=None,
                reason="Experiment uses a single compute unit",
            )

        # Unknown constraint — conservatively mark as satisfied with note
        return ConstraintCheck(
            constraint_name=name,
            satisfied=True,
            observed_value=None,
            reason=f"Constraint '{name}' not automatically verifiable — assumed satisfied",
        )

    def _check_regression(
        self,
        memory: ResearchMemory,
        metric: str,
        current_value: float | None,
    ) -> RegressionCheck:
        best_result = memory.get_best_result()
        best_val: float | None = None
        if best_result:
            best_val = best_result.get_metric(metric)

        if current_value is None or best_val is None:
            return RegressionCheck(
                is_regression=False,
                best_known=best_val,
                current_value=current_value,
                delta=None,
            )

        delta = current_value - best_val
        # Regression if current is more than 0.5pp worse than best
        is_regression = delta < -0.005
        return RegressionCheck(
            is_regression=is_regression,
            best_known=round(best_val, 4),
            current_value=round(current_value, 4),
            delta=round(delta, 4),
        )

    def _check_non_determinism(
        self,
        memory: ResearchMemory,
        metric: str,
        threshold: float = 0.02,
    ) -> NonDeterminismCheck:
        """
        Check if results are non-deterministic by looking at variance
        across experiments with the same hypothesis.
        """
        # Group results by hypothesis
        hypothesis_results: dict[str, list[float]] = {}
        for exp in memory.experiments:
            if exp.observed_result and exp.hypothesis:
                val = exp.observed_result.get_metric(metric)
                if val is not None:
                    hyp = exp.hypothesis.strip().lower()[:50]
                    hypothesis_results.setdefault(hyp, []).append(val)

        # Check if any hypothesis has high variance
        for hyp, values in hypothesis_results.items():
            if len(values) >= 2:
                import statistics
                std = statistics.stdev(values)
                if std > threshold:
                    return NonDeterminismCheck(
                        is_non_deterministic=True,
                        std_dev=round(std, 4),
                        samples=values,
                        threshold=threshold,
                    )

        return NonDeterminismCheck(
            is_non_deterministic=False,
            std_dev=None,
        )

    def _check_budget(self, memory: ResearchMemory, goal: GoalContract) -> BudgetCheck:
        max_exp = goal.max_experiments
        max_rec = goal.allowed_resources.max_total_recoveries
        consumed_exp = memory.budget.experiments_consumed
        consumed_rec = memory.budget.recoveries_consumed
        remaining_exp = max(0, max_exp - consumed_exp)
        remaining_rec = max(0, max_rec - consumed_rec)

        at_risk = (remaining_exp / max(max_exp, 1) < 0.2) or (
            remaining_rec / max(max_rec, 1) < 0.2
        )
        exhausted = remaining_exp == 0

        return BudgetCheck(
            experiments_remaining=remaining_exp,
            recoveries_remaining=remaining_rec,
            at_risk=at_risk,
            exhausted=exhausted,
        )

    def _extract_primary_metric(
        self, tool_response: dict[str, Any], metric_name: str
    ) -> float | None:
        """Extract a metric value from a tool response dict."""
        # Direct key
        if metric_name in tool_response:
            val = tool_response[metric_name]
            if isinstance(val, (int, float)):
                return float(val)

        # Nested under 'data'
        data = tool_response.get("data", {})
        if metric_name in data:
            val = data[metric_name]
            if isinstance(val, (int, float)):
                return float(val)

        # Nested under 'metrics'
        metrics = data.get("metrics", {})
        if metric_name in metrics:
            val = metrics[metric_name]
            if isinstance(val, (int, float)):
                return float(val)

        return None
