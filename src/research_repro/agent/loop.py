"""
Agent Loop — enhanced for Phase 3: full experiment graph + evidence integration.

Changes from Phase 2:
  - EvidenceLedger is now active: paper claims, experiment results, discrepancies
    all get recorded with traceable citations.
  - Hypothesis lifecycle: hypothesis objects are created BEFORE experiments
    and updated AFTER evaluation.
  - Experiment graph: parent-child lineage is tracked per experiment.
  - Discrepancy detection is now immediate and feeds into DiscrepancyAnalyzer.
  - Phase 5: GoalChecker and FailureTaxonomyClassifier are used by SelfEvaluator
    (the loop delegates entirely to the enhanced evaluator).

Budget is still enforced here, not scattered.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .planner import Planner
from .react import ReactController, StepResult
from .state_machine import AgentPhase, StateMachine
from ..config import Config
from ..evaluation.evaluator import Decision, SelfEvaluator
from ..memory.experiment_graph import ExperimentGraph
from ..memory.ledger import EvidenceLedger
from ..memory.models import (
    BudgetState,
    Discrepancy,
    Experiment,
    ExecutionStatus,
    EvaluationStatus,
    ExperimentResult,
    Failure,
    FailureType,
    GoalContract,
    Hypothesis,
    Plan,
    Recovery,
    ResearchMemory,
)
from ..memory.store import MemoryStore
from ..observability.events import EventType
from ..observability.logger import EventLogger
from ..tools.base import ToolRegistry
from ..recovery import RecoveryContext, RecoveryOrchestrator


# ---------------------------------------------------------------------------
# Final Report
# ---------------------------------------------------------------------------


class FinalReport(BaseModel):
    run_id: str
    goal: str
    completed: bool
    total_steps: int
    experiments_run: int
    self_corrections: int
    recovery_failures: int
    best_accuracy: float | None = None
    reported_accuracy: float | None = None
    gap: float | None = None
    unresolved_questions: list[str] = []
    evidence_count: int = 0
    hypothesis_count: int = 0
    discrepancy_count: int = 0
    summary: str = ""

    def display(self) -> str:
        lines = [
            f"Run:              {self.run_id}",
            f"Goal:             {self.goal}",
            f"Completed:        {self.completed}",
            f"Steps:            {self.total_steps}",
            f"Experiments:      {self.experiments_run}",
            f"Self-corrections: {self.self_corrections}",
            f"Evidence items:   {self.evidence_count}",
            f"Hypotheses:       {self.hypothesis_count}",
            f"Discrepancies:    {self.discrepancy_count}",
            f"Best accuracy:    {self.best_accuracy}",
            f"Reported:         {self.reported_accuracy}",
            f"Gap:              {self.gap}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent Loop
# ---------------------------------------------------------------------------


class AgentLoop:
    """
    Enhanced agent loop with full experiment graph + evidence integration.
    """

    MAX_STEPS_SAFETY = 50

    def __init__(
        self,
        goal: GoalContract,
        run_id: str,
        config: Config,
        registry: ToolRegistry,
    ) -> None:
        self.goal = goal
        self.run_id = run_id
        self.config = config
        self.registry = registry

        # Directories
        self.run_dir = config.runs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Core infrastructure
        self.logger = EventLogger(run_id, self.run_dir)
        self.memory_store = MemoryStore(self.run_dir)
        self.ledger = EvidenceLedger(self.run_dir)       # Phase 3: evidence ledger
        self.state_machine = StateMachine()

        # LLM components (different system prompts — same model)
        self.planner = Planner(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.llm_base_url,
        )
        self.evaluator = SelfEvaluator(
            api_key=config.openai_api_key,
            model=config.openai_model,
            base_url=config.llm_base_url,
        )

        # ReAct controller
        self.react = ReactController(
            planner=self.planner,
            evaluator=self.evaluator,
            registry=self.registry,
            logger=self.logger,
        )
        self.recovery = RecoveryOrchestrator(
            max_total_recoveries=goal.allowed_resources.max_total_recoveries,
            max_recoveries_per_subtask=goal.allowed_resources.max_recoveries_per_subtask,
        )

        # Initialize working memory
        self.memory = ResearchMemory(
            goal=goal,
            run_id=run_id,
            current_phase=AgentPhase.INITIALIZING.value,
            budget=BudgetState(),
        )
        self.memory_store.save(self.memory)

        # Phase 3: experiment graph (wraps memory.experiments)
        self.graph = ExperimentGraph(self.memory.experiments)

        # Track current hypothesis for lifecycle management
        self._current_hypothesis_id: str | None = None

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self) -> FinalReport:
        """Run the agent loop to completion."""
        self.logger.log(EventType.GOAL_SET, goal=self.goal.model_dump(mode="json"))
        self._create_initial_plan()
        self._transition(AgentPhase.LITERATURE_DISCOVERY)

        self_corrections = 0
        recovery_failures = 0
        step_count = 0

        while not self.state_machine.is_terminal:
            step_count += 1
            if step_count > self.MAX_STEPS_SAFETY:
                self.logger.log(
                    EventType.BUDGET_EXHAUSTED,
                    reason="safety_step_limit",
                    steps=step_count,
                )
                break

            if not self.memory.budget.can_run_experiment(self.goal.max_experiments):
                self.logger.log(
                    EventType.BUDGET_EXHAUSTED,
                    reason="max_experiments",
                    consumed=self.memory.budget.experiments_consumed,
                )
                break

            try:
                result = self.react.step(self.memory)
                self.memory.step_count += 1
                self._apply_step_result(result)
                self.memory_store.save(self.memory)

            except Exception as exc:
                self.logger.log(
                    EventType.FAILURE_DETECTED,
                    type="unexpected_exception",
                    error=str(exc),
                )
                break

            # Route on decision
            if result.decision == Decision.GOAL_ACHIEVED:
                self._transition(AgentPhase.GENERATING_REPORT)
                break
            elif result.decision == Decision.DIAGNOSE_AND_RECOVER:
                self_corrections += 1
                recovered = self._handle_failure(result)
                if not recovered:
                    recovery_failures += 1
            elif result.decision == Decision.TERMINATE:
                break

        return self._build_report(self_corrections, recovery_failures)

    # ── Step result application (Phase 3: full integration) ─────────────────

    def _apply_step_result(self, result: StepResult) -> None:
        """
        Apply a step result to ResearchMemory with full Phase 3 integration:
          - Evidence recording for paper/experiment claims
          - Hypothesis lifecycle (create before, update after)
          - Experiment graph node addition
          - Discrepancy detection with DiscrepancyAnalyzer
        """
        eval_ = result.evaluation

        # ── Record failures ───────────────────────────────────────────────
        if eval_.failure_type is not None:
            failure = Failure(
                failure_type=eval_.failure_type,
                description=eval_.rationale,
                step_index=result.step_index,
            )
            self.memory.failures.append(failure)
            self.logger.log(
                EventType.FAILURE_DETECTED,
                failure_type=eval_.failure_type.value,
                description=eval_.rationale,
                step=result.step_index,
            )

        # ── Paper-related tools: record evidence ──────────────────────────
        if result.tool_name in ("retrieve_paper", "extract_methodology"):
            self._record_paper_evidence(result)

        # ── Experiment execution ──────────────────────────────────────────
        if result.tool_name == "run_experiment" and result.tool_response.success:
            self._handle_experiment_result(result)

        # ── Comparison: discrepancy detection ─────────────────────────────
        if result.tool_name == "compare_results" and result.tool_response.success:
            self._handle_comparison_result(result)

        # ── Observation recording ─────────────────────────────────────────
        if result.tool_response.success and result.tool_name not in (
            "run_experiment", "compare_results"
        ):
            self.ledger.record_observation(
                memory=self.memory,
                claim=result.reasoning.hypothesis,
                step_index=result.step_index,
                tool_name=result.tool_name,
            )

        # ── Hypothesis lifecycle: update status after evaluation ───────────
        self._update_hypothesis_status(result)

    # ── Evidence recording ────────────────────────────────────────────────────

    def _record_paper_evidence(self, result: StepResult) -> None:
        """Record evidence claims from paper-reading tool calls."""
        data = result.tool_response.data
        paper_id = data.get("id") or result.tool_args.get("paper_id", "unknown_paper")

        if result.tool_name == "retrieve_paper":
            title = data.get("title", "paper")
            self.ledger.record_paper_claim(
                memory=self.memory,
                claim=f"Retrieved paper: '{title}'",
                paper_id=paper_id,
                section="title",
            )
            # Record abstract evidence
            abstract = data.get("abstract", "")
            if abstract:
                self.ledger.record_paper_claim(
                    memory=self.memory,
                    claim=f"Paper abstract: {abstract[:100]}...",
                    paper_id=paper_id,
                    section="abstract",
                )
            # Store selected paper in memory
            from ..memory.models import Paper
            self.memory.selected_paper = Paper(
                id=paper_id,
                title=data.get("title", ""),
                authors=data.get("authors", []),
                abstract=data.get("abstract", ""),
                url=data.get("url", ""),
            )

        elif result.tool_name == "extract_methodology":
            from ..memory.models import Methodology
            reported = data.get("reported_metrics", {})
            for metric, value in reported.items():
                self.ledger.record_paper_claim(
                    memory=self.memory,
                    claim=f"Paper reports {metric}={value}",
                    paper_id=paper_id,
                    section="results",
                )
                self.logger.log(EventType.EVIDENCE_RECORDED, metric=metric, value=value)

            # Store methodology
            self.memory.methodology = Methodology(
                paper_id=paper_id,
                dataset=data.get("dataset", ""),
                model_architecture=data.get("model_architecture", ""),
                hyperparameters=data.get("hyperparameters", {}),
                preprocessing_steps=data.get("preprocessing", []),
                training_details=data.get("training_details", {}),
                reported_metrics=reported,
            )

    def _handle_experiment_result(self, result: StepResult) -> None:
        """Record a completed experiment into memory and evidence ledger."""
        self.memory.budget.consume_experiment()
        data = result.tool_response.data
        exp_id = data.get("experiment_id", f"exp_{uuid.uuid4().hex[:8]}")

        # Create typed Experiment object
        exp = Experiment(
            id=exp_id,
            parent_id=result.tool_args.get("parent_experiment_id"),
            hypothesis=result.reasoning.hypothesis,
            parameters=result.tool_args.get("parameters", {}),
            expected_result={"accuracy": self.goal.target_value or 0},
            git_commit=data.get("git_commit"),
        )

        # Record observed result
        exp.observed_result = ExperimentResult(
            accuracy=data.get("accuracy"),
            f1=data.get("f1"),
            loss=data.get("loss"),
            runtime_seconds=data.get("runtime_seconds"),
            additional_metrics={
                k: v for k, v in data.items()
                if k not in ("accuracy", "f1", "loss", "runtime_seconds", "experiment_id", "git_commit")
                and isinstance(v, (int, float))
            },
        )
        exp.execution_status = ExecutionStatus.SUCCESS
        exp.evaluation_status = (
            EvaluationStatus.POSITIVE
            if result.evaluation.goal_progress.value == "positive"
            else EvaluationStatus.NEGATIVE
        )
        exp.completed_at = datetime.utcnow()

        # Link to current hypothesis
        if self._current_hypothesis_id:
            for h in self.memory.hypotheses:
                if h.id == self._current_hypothesis_id:
                    h.experiment_id = exp_id
                    break

        self.memory.experiments.append(exp)
        self.graph.add_node(exp)
        self.memory.update_best(self.goal.primary_metric)

        # Record experiment evidence
        acc = exp.observed_result.accuracy
        if acc is not None:
            self.ledger.record_experiment_result(
                memory=self.memory,
                claim=f"Experiment '{exp_id}' achieved accuracy={acc:.4f}",
                experiment_id=exp_id,
                git_commit=exp.git_commit or "",
            )

        self.logger.log(EventType.EXPERIMENT_STARTED, experiment_id=exp_id)
        self.logger.log(
            EventType.EXPERIMENT_COMPLETED,
            experiment_id=exp_id,
            result=exp.observed_result.model_dump(mode="json"),
        )

    def _handle_comparison_result(self, result: StepResult) -> None:
        """Handle compare_results tool output — record discrepancy if found."""
        data = result.tool_response.data
        if data.get("status") == "DISCREPANCY":
            reported = data.get("reported", 0)
            observed = data.get("observed", 0)
            metric = self.goal.primary_metric

            disc = Discrepancy.from_values(metric=metric, expected=reported, observed=observed)
            self.memory.discrepancies.append(disc)

            self.ledger.record_discrepancy(
                memory=self.memory,
                metric=metric,
                expected=reported,
                observed=observed,
                experiment_id=result.tool_args.get("experiment_id", "unknown"),
            )

            self.logger.log(
                EventType.DISCREPANCY_DETECTED,
                metric=metric,
                expected=reported,
                observed=observed,
                magnitude=disc.magnitude,
                gap_abs=data.get("absolute_difference"),
                gap_rel_pct=data.get("relative_difference_pct"),
            )

    def _update_hypothesis_status(self, result: StepResult) -> None:
        """Update the hypothesis status based on evaluation."""
        if self._current_hypothesis_id:
            eval_status = result.evaluation.hypothesis_status.value
            for h in self.memory.hypotheses:
                if h.id == self._current_hypothesis_id:
                    h.based_on_evidence.append(f"step_{result.step_index}:{eval_status}")
                    break

    # ── Hypothesis lifecycle ─────────────────────────────────────────────────

    def _create_hypothesis(self, hypothesis_text: str) -> str:
        """Create a hypothesis object and track it. Returns hypothesis ID."""
        h = Hypothesis(
            statement=hypothesis_text,
            based_on_evidence=[],
        )
        self.memory.hypotheses.append(h)
        self._current_hypothesis_id = h.id
        return h.id

    # ── Failure handling ─────────────────────────────────────────────────────

    def _handle_failure(self, result: StepResult) -> bool:
        """Dispatch a concrete recovery strategy and preserve its audit trail."""
        if not self.memory.failures or result.evaluation.failure_type is None:
            return False
        failure = self.memory.failures[-1]
        failure_type = result.evaluation.failure_type.value
        self.logger.log(EventType.RECOVERY_STARTED, failure_type=failure_type)
        context = RecoveryContext(
            memory=self.memory,
            failure=failure,
            tool_args=result.tool_args,
            tool_response=result.tool_response.model_dump(mode="json"),
        )
        # Provide checkpointer, independent evaluator, and schema repair callbacks
        try:
            context.checkpointer = self.registry.get("run_experiment").checkpointer
        except (KeyError, AttributeError):
            context.checkpointer = None

        if self.registry.has("run_independent_evaluation"):
            eval_tool = self.registry.get("run_independent_evaluation")
            context.independent_evaluator = lambda args: eval_tool.execute(
                eval_tool.request_model(**args) if isinstance(args, dict) else args
            ).data

        context.schema_repair = lambda args, errors: self._repair_schema_args(result.tool_name, args, errors)

        outcome = self.recovery.recover(context, subtask_id=result.tool_name)
        self.memory_store.save(self.memory)
        self.logger.log(EventType.RECOVERY_COMPLETED, success=outcome.success, strategy=outcome.strategy, action=outcome.action_taken)
        if not outcome.success:
            self.logger.log(EventType.BUDGET_EXHAUSTED if outcome.strategy == "budget_exhausted" else EventType.FAILURE_DETECTED, reason=outcome.unresolved or outcome.action_taken)
        return outcome.success

    def _repair_schema_args(self, tool_name: str, args: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
        """Auto-repair common schema errors using Pydantic field error feedback."""
        repaired = copy.deepcopy(args)
        for err in errors:
            field = err.get("field")
            err_type = err.get("type", "")
            if not field:
                continue
            if "missing" in err_type:
                # Provide standard default based on field name
                if "id" in field:
                    repaired[field] = f"exp_{int(time.time())}"
                elif "metric" in field:
                    repaired[field] = self.goal.primary_metric
                elif "threshold" in field:
                    repaired[field] = 0.01
                elif "timeout" in field:
                    repaired[field] = 300
            elif "int_parsing" in err_type or "int_type" in err_type:
                try:
                    repaired[field] = int(float(repaired[field]))
                except (ValueError, TypeError):
                    repaired[field] = 42
            elif "float_parsing" in err_type or "float_type" in err_type:
                try:
                    repaired[field] = float(repaired[field])
                except (ValueError, TypeError):
                    repaired[field] = 0.0
        return repaired

    # ── Utilities ────────────────────────────────────────────────────────────

    def _create_initial_plan(self) -> None:
        plan = Plan(
            steps=[
                "Search literature for relevant papers",
                "Select best paper matching goal",
                "Extract experimental methodology",
                "Plan baseline reproduction experiment",
                "Run baseline experiment",
                "Compare results with reported values",
                "Diagnose any discrepancies",
                "Apply targeted fixes and re-run",
                "Generate verified final report",
            ],
            current_step_index=0,
            version=1,
        )
        self.memory.current_plan = plan
        self.memory_store.save(self.memory)
        self.logger.log(EventType.PLAN_CREATED, steps=plan.steps, version=1)

    def _transition(self, to: AgentPhase) -> None:
        from_phase = self.state_machine.current
        self.state_machine.transition(to)
        self.memory.current_phase = to.value
        self.memory_store.save(self.memory)
        self.logger.log(
            EventType.PHASE_TRANSITION,
            from_phase=from_phase.value,
            to_phase=to.value,
        )

    def _build_report(self, self_corrections: int, recovery_failures: int) -> FinalReport:
        best_result = self.memory.get_best_result()
        best_acc: float | None = None
        if best_result:
            best_acc = best_result.get_metric(self.goal.primary_metric)
        gap = None
        if best_acc is not None and self.goal.target_value is not None:
            gap = round(abs(self.goal.target_value - best_acc), 4)

        report = FinalReport(
            run_id=self.run_id,
            goal=self.goal.objective,
            completed=self.state_machine.current == AgentPhase.COMPLETED,
            total_steps=self.memory.step_count,
            experiments_run=self.memory.budget.experiments_consumed,
            self_corrections=self_corrections,
            recovery_failures=recovery_failures,
            best_accuracy=best_acc,
            reported_accuracy=self.goal.target_value,
            gap=gap,
            unresolved_questions=self.memory.unresolved_questions,
            evidence_count=len(self.memory.evidence),
            hypothesis_count=len(self.memory.hypotheses),
            discrepancy_count=len(self.memory.discrepancies),
            summary=(
                f"Ran {self.memory.budget.experiments_consumed} experiments, "
                f"{self_corrections} self-corrections, "
                f"best {self.goal.primary_metric}: {best_acc}"
            ),
        )

        self.logger.log(EventType.FINAL_REPORT, report=report.model_dump(mode="json"))
        self.memory_store.save(self.memory)
        return report
