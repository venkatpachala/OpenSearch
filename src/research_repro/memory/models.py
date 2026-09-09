"""
Core data models for the ResearchRepro agent.

All state is captured in ResearchMemory — a typed, serializable snapshot
of everything the agent knows about the current run. This is NOT a chat
history; it's a structured research state object.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum, EnumMeta
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class _FailureTypeMeta(EnumMeta):
    """Keep the original six-value public taxonomy iteration compatible.

    RESOURCE_FAILURE is an additive extension available by name, while the
    original Heva six remain the canonical benchmark set.
    """
    def __iter__(cls):
        return (member for member in super().__iter__() if member.name != "RESOURCE_FAILURE")


class FailureType(str, Enum, metaclass=_FailureTypeMeta):
    """The six causally distinct failure categories."""

    TOOL_CRASH = "tool_crash"
    """Tool process crashed or returned a non-zero exit code."""

    TOOL_SCHEMA_ERROR = "tool_schema_error"
    """Tool received arguments that failed Pydantic schema validation."""

    RESULT_INCONSISTENCY = "result_inconsistency"
    """Training reports one metric; independent evaluator reports a different one."""

    GOAL_DRIFT = "goal_drift"
    """Result improves primary metric but violates one or more constraints."""

    REGRESSION = "regression"
    """New experiment performs worse than the current best known result."""

    NON_DETERMINISM = "non_determinism"
    """Repeated runs with the same config produce meaningfully different results."""

    RESOURCE_FAILURE = "resource_failure"
    """OOM, unavailable accelerator, disk exhaustion, or hard resource limit."""


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    POSITIVE = "positive"
    NEGATIVE = "negative"
    INCONCLUSIVE = "inconclusive"


class HypothesisLifecycle(str, Enum):
    PROPOSED = "proposed"
    TESTED = "tested"
    SUPPORTED = "supported"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class ReproductionVerdict(str, Enum):
    CONFIRMED = "confirmed"
    INCONCLUSIVE = "inconclusive"
    REFUTED = "refuted"


# ---------------------------------------------------------------------------
# Budget & Constraints
# ---------------------------------------------------------------------------


class ResourceBudget(BaseModel):
    """Hard limits on experiments and recovery attempts."""

    max_experiments: int = 8
    max_recoveries_per_subtask: int = 2
    max_total_recoveries: int = 6
    timeout_seconds: int = 3600


class Constraint(BaseModel):
    name: str
    description: str


class MetricCriterion(BaseModel):
    """Explicit metric semantics; avoids treating 0.5 as 0.5 percentage points."""

    metric: str
    direction: Literal["maximize", "minimize", "match"] = "match"
    target: float | None = None
    tolerance: float | None = None
    minimum: float | None = None
    maximum: float | None = None

    def satisfied(self, observed: float | None) -> bool:
        if observed is None:
            return False
        if self.direction == "maximize":
            min_val = self.minimum if self.minimum is not None else ((self.target - (self.tolerance or 0.0)) if self.target is not None else None)
            return min_val is None or observed >= min_val
        if self.direction == "minimize":
            max_val = self.maximum if self.maximum is not None else ((self.target + (self.tolerance or 0.0)) if self.target is not None else None)
            return max_val is None or observed <= max_val
        if self.direction == "match":
            return self.target is not None and self.tolerance is not None and abs(observed - self.target) <= self.tolerance
        if self.minimum is not None and observed < self.minimum:
            return False
        if self.maximum is not None and observed > self.maximum:
            return False
        return True


# ---------------------------------------------------------------------------
# Goal Contract — IMMUTABLE after creation
# ---------------------------------------------------------------------------


class GoalContract(BaseModel):
    """
    The user's success criteria, expressed as a structured contract.
    This object is NEVER modified during a run. It is the authoritative
    definition of what constitutes success.
    """

    objective: str
    primary_metric: str = "accuracy"
    target_value: float | None = None
    success_threshold: float = 1.0
    """Acceptable deviation from target_value (in same units)."""
    constraints: list[Constraint] = Field(default_factory=list)
    criterion: MetricCriterion | None = None
    max_experiments: int = 8
    allowed_resources: ResourceBudget = Field(default_factory=ResourceBudget)

    def is_achieved(self, observed: float) -> bool:
        """Check if an observed metric value satisfies the goal."""
        if self.criterion is not None:
            return self.criterion.satisfied(observed)
        if self.target_value is None:
            return False
        return abs(self.target_value - observed) <= self.success_threshold

    def constraints_hold(self, observation: dict[str, Any] | None = None) -> bool:
        """Return False if a named constraint is observably violated.

        Metric success alone is not a contract pass. Unknown constraints
        (no supporting observation) are treated as holding.
        """
        data = observation or {}
        nested = data.get("additional_metrics") if isinstance(data.get("additional_metrics"), dict) else {}
        for constraint in self.constraints:
            name = constraint.name.lower()
            if "latency" in name:
                latency = data.get("latency_ms")
                if latency is None:
                    latency = nested.get("latency_ms")
                if latency is not None and float(latency) > 100.0:
                    return False
        return True

    def meets_contract(self, observed: float | None, observation: dict[str, Any] | None = None) -> bool:
        """True only when the primary metric and every constraint hold."""
        if observed is None:
            return False
        return self.is_achieved(observed) and self.constraints_hold(observation)


# ---------------------------------------------------------------------------
# Budget State — mutable, tracked per run
# ---------------------------------------------------------------------------


class BudgetState(BaseModel):
    experiments_consumed: int = 0
    recoveries_consumed: int = 0
    recoveries_per_subtask: dict[str, int] = Field(default_factory=dict)

    def can_run_experiment(self, max_experiments: int) -> bool:
        return self.experiments_consumed < max_experiments

    def can_recover(self, max_total: int) -> bool:
        return self.recoveries_consumed < max_total

    def can_recover_subtask(self, subtask_id: str, max_per_subtask: int) -> bool:
        return self.recoveries_per_subtask.get(subtask_id, 0) < max_per_subtask

    def consume_experiment(self) -> None:
        self.experiments_consumed += 1

    def consume_recovery(self, subtask_id: str | None = None) -> None:
        self.recoveries_consumed += 1
        if subtask_id:
            self.recoveries_per_subtask[subtask_id] = (
                self.recoveries_per_subtask.get(subtask_id, 0) + 1
            )


# ---------------------------------------------------------------------------
# Research artifacts
# ---------------------------------------------------------------------------


class Paper(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    url: str = ""
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class Methodology(BaseModel):
    """Extracted experimental details from a paper."""

    paper_id: str
    dataset: str = ""
    model_architecture: str = ""
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    preprocessing_steps: list[str] = Field(default_factory=list)
    training_details: dict[str, Any] = Field(default_factory=dict)
    reported_metrics: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experiment lifecycle
# ---------------------------------------------------------------------------


class ExperimentResult(BaseModel):
    """Observed metrics from a completed experiment."""

    accuracy: float | None = None
    f1: float | None = None
    loss: float | None = None
    runtime_seconds: float | None = None
    additional_metrics: dict[str, float] = Field(default_factory=dict)

    def get_metric(self, metric_name: str) -> float | None:
        """Return a metric by name, checking standard fields first."""
        standard = {"accuracy": self.accuracy, "f1": self.f1, "loss": self.loss}
        return standard.get(metric_name) or self.additional_metrics.get(metric_name)


class Experiment(BaseModel):
    """
    A single experiment node in the experiment graph.
    Experiments form a tree rooted at the baseline.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: str | None = None
    hypothesis: str = ""
    code_changes: list[str] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    expected_result: dict[str, Any] = Field(default_factory=dict)
    observed_result: ExperimentResult | None = None
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    evaluation_status: EvaluationStatus = EvaluationStatus.PENDING
    failure_id: str | None = None
    recovery_id: str | None = None
    git_commit: str | None = None
    config_fingerprint: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Reasoning artifacts
# ---------------------------------------------------------------------------


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    statement: str
    based_on_evidence: list[str] = Field(default_factory=list)
    experiment_id: str | None = None
    status: HypothesisLifecycle = HypothesisLifecycle.PROPOSED


class Discrepancy(BaseModel):
    """A detected gap between expected and observed values."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    metric: str
    expected: float
    observed: float
    magnitude: float
    possible_causes: list[str] = Field(default_factory=list)

    @classmethod
    def from_values(cls, metric: str, expected: float, observed: float) -> "Discrepancy":
        return cls(
            metric=metric,
            expected=expected,
            observed=observed,
            magnitude=abs(expected - observed),
        )


class Evidence(BaseModel):
    """A traceable claim backed by a source."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    claim: str
    source_type: str  # "paper" | "experiment" | "observation"
    source_id: str
    location: str = ""  # e.g. "page=7, section='Training Details'"
    recorded_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Failure & Recovery
# ---------------------------------------------------------------------------


class Failure(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    failure_type: FailureType
    description: str
    experiment_id: str | None = None
    step_index: int | None = None
    recovered: bool = False


class Recovery(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    failure_id: str
    strategy: str
    action_taken: str
    success: bool = False
    diagnosis: str = ""
    evidence_used: list[str] = Field(default_factory=list)
    expected_observation: str = ""
    plan_change: str = ""


class ReproductionAssessment(BaseModel):
    metric_success: bool = False
    methodology_aligned: bool | None = None
    independently_verified: bool | None = None
    verdict: ReproductionVerdict = ReproductionVerdict.INCONCLUSIVE
    rationale: str = ""


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


class Plan(BaseModel):
    steps: list[str] = Field(default_factory=list)
    current_step_index: int = 0
    version: int = 1

    def advance(self) -> bool:
        """Advance to next step. Returns False if already at end."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            return True
        return False

    @property
    def current_step(self) -> str | None:
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None


# ---------------------------------------------------------------------------
# Master State Object
# ---------------------------------------------------------------------------


class ResearchMemory(BaseModel):
    """
    The complete, typed state of a research reproduction run.

    This is NOT a chat history. Every field has semantic meaning.
    The agent reasons over this structure, not over raw messages.
    """

    # Identity
    run_id: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    step_count: int = 0

    # Immutable goal
    goal: GoalContract

    # Mutable phase + plan
    current_phase: str = "initializing"
    current_plan: Plan = Field(default_factory=Plan)
    plan_version: int = 1

    # Research artifacts (populated during run)
    selected_paper: Paper | None = None
    methodology: Methodology | None = None
    baseline: ExperimentResult | None = None

    # Discovered papers from literature search (populated after search_literature calls)
    discovered_papers: list[dict] = Field(default_factory=list)

    # Experiment lineage
    experiments: list[Experiment] = Field(default_factory=list)
    best_experiment_id: str | None = None

    # Reasoning trail
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    discrepancies: list[Discrepancy] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)

    # Failure + recovery history
    failures: list[Failure] = Field(default_factory=list)
    recoveries: list[Recovery] = Field(default_factory=list)

    # Open questions the agent couldn't resolve
    unresolved_questions: list[str] = Field(default_factory=list)
    reproduction: ReproductionAssessment = Field(default_factory=ReproductionAssessment)

    # Budget tracking
    budget: BudgetState = Field(default_factory=BudgetState)

    # Planner no-progress: repeated rejected configurations
    no_progress_count: int = 0
    blocked_proposals: list[dict[str, Any]] = Field(default_factory=list)

    # ---------- Convenience helpers ----------

    def get_best_result(self) -> ExperimentResult | None:
        if not self.best_experiment_id:
            return None
        for exp in self.experiments:
            if exp.id == self.best_experiment_id:
                return exp.observed_result
        return None

    def update_best(self, metric: str) -> None:
        """Pick the best constraint-satisfying experiment. Illegal runs stay off best."""
        best_val: float | None = None
        self.best_experiment_id = None
        for exp in self.experiments:
            if not exp.observed_result:
                continue
            val = exp.observed_result.get_metric(metric)
            if val is None:
                continue
            observation = {
                metric: val,
                "latency_ms": exp.observed_result.additional_metrics.get("latency_ms"),
                "additional_metrics": exp.observed_result.additional_metrics,
            }
            if not self.goal.constraints_hold(observation):
                continue
            if best_val is None or val > best_val:
                best_val = val
                self.best_experiment_id = exp.id

    def recent_failures(self, n: int = 3) -> list[Failure]:
        return self.failures[-n:]

    def recent_experiments(self, n: int = 3) -> list[Experiment]:
        return self.experiments[-n:]

    def recent_discoveries(self, n: int = 5) -> list[dict]:
        return self.discovered_papers[-n:]
