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

import copy
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .planner import Planner, PlannerOutput
from .planning_context import (
    apply_progress,
    assess_proposal,
    build_planning_evidence,
    canonicalize_tool_args,
    classify_progress,
    configuration_fingerprint,
    criterion_status_map,
    experiment_parameters,
    normalize_parameters,
    record_blocked_proposal,
    record_configuration_outcome,
    repair_from_argparse_error,
)
from .react import ReactController, StepResult
from .state_machine import AgentPhase, StateMachine
from ..config import Config
from ..evaluation.evaluator import (
    ConstraintStatus,
    Decision,
    EvaluationResult,
    GoalProgress,
    HypothesisStatus,
    SelfEvaluator,
)
from ..memory.experiment_graph import ExperimentGraph
from ..memory.ledger import EvidenceLedger
from ..memory.models import (
    BudgetState,
    ConfigurationOutcome,
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
    HypothesisLifecycle,
    ReproductionAssessment,
    ReproductionVerdict,
)
from ..memory.store import MemoryStore
from ..observability.events import EventType
from ..observability.logger import EventLogger
from ..tools.base import ToolRegistry, ToolResponse
from ..recovery import RecoveryContext, RecoveryOrchestrator
from .strategy import AgentStrategy, SelfCorrectingStrategy


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

    MAX_STEPS_SAFETY = 120

    def __init__(
        self,
        goal: GoalContract,
        run_id: str,
        config: Config,
        registry: ToolRegistry,
        strategy: AgentStrategy | None = None,
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
            proposal_guard=self._guard_proposal,
        )
        self.recovery = RecoveryOrchestrator(
            max_total_recoveries=goal.allowed_resources.max_total_recoveries,
            max_recoveries_per_subtask=goal.allowed_resources.max_recoveries_per_subtask,
        )
        try:
            checkpointer = self.registry.get("run_experiment").checkpointer
        except (KeyError, AttributeError):
            checkpointer = None
        independent_evaluator = None
        if "run_independent_evaluation" in self.registry:
            eval_tool = self.registry.get("run_independent_evaluation")
            independent_evaluator = lambda args: eval_tool.execute(
                eval_tool.request_model(**args) if isinstance(args, dict) else args
            ).data
        self.strategy: AgentStrategy = strategy or SelfCorrectingStrategy(
            self.recovery,
            checkpointer,
            independent_evaluator=independent_evaluator,
            schema_repair=lambda args, errors: self._repair_schema_args("", args, errors),
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
        self._pending_retry: StepResult | None = None
        self._last_strategy_decision = None

    def _guard_proposal(
        self,
        memory: ResearchMemory,
        planner_output: PlannerOutput,
        step_index: int,
        record: bool = True,
    ) -> StepResult | None:
        """Block silent repeats of rejected configs. Does not run the experiment."""
        assessment = assess_proposal(memory, planner_output.selected_tool, planner_output.tool_arguments)
        if not assessment.blocked:
            return None
        if record:
            record_blocked_proposal(memory, assessment, planner_output.tool_arguments)
            self.logger.log(
                EventType.PROPOSAL_BLOCKED,
                step=step_index,
                reason=assessment.reason,
                fingerprint=assessment.fingerprint,
                block_code=assessment.block_code,
                previous_experiment=assessment.previous_experiment,
                previous_outcome=assessment.previous_outcome,
                terminal=assessment.terminal,
                streak=memory.no_progress_count,
            )
            self.logger.log(
                EventType.NO_PROGRESS,
                step=step_index,
                reason=assessment.reason,
                fingerprint=assessment.fingerprint,
                terminal=assessment.terminal,
                streak=memory.no_progress_count,
                terminate_reason=memory.terminate_reason,
            )
            if assessment.terminal:
                self.logger.log(
                    EventType.CAMPAIGN_TERMINATED,
                    step=step_index,
                    terminate_reason=memory.terminate_reason or "no_progress",
                    completed=False,
                    streak=memory.no_progress_count,
                )
        else:
            memory.blocked_proposals.append({
                "fingerprint": assessment.fingerprint,
                "configuration": planner_output.tool_arguments.get("parameters") or {},
                "reason": assessment.reason,
            })
        evaluation = EvaluationResult(
            execution_success=False,
            output_valid=False,
            metric_valid=False,
            goal_progress=GoalProgress.NEGATIVE,
            constraint_status=ConstraintStatus.ALL_MET,
            hypothesis_status=HypothesisStatus.REJECTED,
            failure_type=None,
            decision=Decision.TERMINATE if (record and assessment.terminal) else Decision.CONTINUE,
            confidence=1.0,
            rationale=assessment.reason,
        )
        return StepResult(
            step_index=step_index,
            reasoning=planner_output.reasoning,
            tool_name=planner_output.selected_tool,
            tool_args=planner_output.tool_arguments,
            tool_response=ToolResponse.fail(assessment.reason),
            evaluation=evaluation,
            decision=evaluation.decision,
        )

    # ── Public API ───────────────────────────────────────────────────────────

    def run(self) -> FinalReport:
        """Run the agent loop to completion."""
        self.logger.log(EventType.GOAL_SET, goal=self.goal.model_dump(mode="json"))
        self._create_initial_plan()  # may pre-transition for direct goals
        # Only go to LITERATURE_DISCOVERY if still in INITIALIZING phase
        if self.state_machine.can_transition(AgentPhase.LITERATURE_DISCOVERY):
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

            if self.memory.terminate_reason == "no_progress":
                self.logger.log(
                    EventType.CAMPAIGN_TERMINATED,
                    terminate_reason="no_progress",
                    completed=False,
                    streak=self.memory.no_progress_count,
                )
                break

            if not self.memory.budget.can_run_experiment(self.goal.max_experiments):
                self.memory.terminate_reason = self.memory.terminate_reason or "max_experiments"
                self.logger.log(
                    EventType.BUDGET_EXHAUSTED,
                    reason="max_experiments",
                    consumed=self.memory.budget.experiments_consumed,
                )
                self.logger.log(
                    EventType.CAMPAIGN_TERMINATED,
                    terminate_reason="max_experiments",
                    completed=False,
                )
                break

            try:
                result = (
                    self.react.repeat(self.memory, self._pending_retry)
                    if self._pending_retry is not None
                    else self.react.step(self.memory)
                )
                self._pending_retry = None
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
                verified = self._maybe_independent_evaluation(result)
                if verified is False:
                    # Independent eval disagreed — treat as inconsistency, not success.
                    self_corrections += 1
                    recovered = self._handle_failure(result)
                    if not recovered:
                        recovery_failures += 1
                    continue
                self.logger.log(
                    EventType.GOAL_ACHIEVED,
                    metric=self.goal.primary_metric,
                    target=self.goal.target_value,
                    source="evaluator",
                    independently_verified=self.memory.reproduction.independently_verified,
                )
                self._try_transition(AgentPhase.GENERATING_REPORT)
                self._try_transition(AgentPhase.COMPLETED)
                break
            elif result.decision == Decision.DIAGNOSE_AND_RECOVER:
                self_corrections += 1
                recovered = self._handle_failure(result)
                if not recovered:
                    recovery_failures += 1
                elif self._last_strategy_decision and self._last_strategy_decision.action == "retry":
                    repaired = self._last_strategy_decision.repaired_tool_args
                    if repaired:
                        result.tool_args = repaired
                    self._pending_retry = result
                if self._last_strategy_decision and self._last_strategy_decision.action == "terminate":
                    break
            elif result.decision == Decision.TERMINATE:
                self.memory.terminate_reason = self.memory.terminate_reason or "evaluator_terminate"
                self.logger.log(
                    EventType.CAMPAIGN_TERMINATED,
                    terminate_reason=self.memory.terminate_reason,
                    completed=False,
                )
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
                experiment_id=result.tool_args.get("experiment_id"),
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
        elif result.tool_name == "run_experiment" and not result.tool_response.success:
            self._handle_failed_experiment(result)

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

        # ── Phase and plan auto-progression based on tool success ─────────
        self._auto_advance_phase(result)

        # ── Hypothesis lifecycle: update status after evaluation ───────────
        self._update_hypothesis_status(result)
        self._update_reproduction_assessment(result)

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
        params = normalize_parameters(experiment_parameters(result.tool_args) or result.tool_args.get("parameters") or {})
        evidence = build_planning_evidence(self.memory)
        self._create_hypothesis(result.reasoning.hypothesis)
        parent_id = result.tool_args.get("parent_experiment_id") or self.memory.best_experiment_id
        if parent_id is None and self.memory.experiments:
            parent_id = self.memory.experiments[-1].id
        expected: dict[str, Any] = {}
        if self.goal.target_value is not None:
            expected[self.goal.primary_metric] = self.goal.target_value
        if any("latency" in c.name.lower() for c in self.goal.constraints):
            expected["latency_ms"] = 100.0
        expected["prediction"] = result.reasoning.intended_action
        trigger = evidence.blocking_criterion or (
            result.evaluation.failure_type.value if result.evaluation.failure_type else "initial"
        )
        reason = evidence.required_change or result.reasoning.intended_action
        if not self.memory.experiments:
            parent_id = None
            reason = reason or "baseline"
            trigger = "baseline"

        # Create typed Experiment object
        exp = Experiment(
            id=exp_id,
            parent_id=parent_id,
            hypothesis=result.reasoning.hypothesis,
            reason=reason,
            trigger=trigger,
            parameters=params,
            expected_result=expected,
            git_commit=data.get("git_commit"),
            config_fingerprint=configuration_fingerprint(result.tool_args),
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
        rejected = self._is_rejected_experiment(result)
        exp.evaluation_status = (
            EvaluationStatus.NEGATIVE
            if rejected
            else EvaluationStatus.POSITIVE
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

        observation = {
            self.goal.primary_metric: exp.observed_result.accuracy,
            "accuracy": exp.observed_result.accuracy,
            "latency_ms": exp.observed_result.additional_metrics.get("latency_ms"),
            "additional_metrics": exp.observed_result.additional_metrics,
        }
        status_map = criterion_status_map(self.goal, observation)
        if rejected:
            outcome = ConfigurationOutcome.CONSTRAINT_VIOLATION
        elif result.evaluation.decision == Decision.GOAL_ACHIEVED or (
            result.evaluation.constraint_status.value == "all_met"
            and self.goal.meets_contract(exp.observed_result.get_metric(self.goal.primary_metric), observation)
        ):
            outcome = ConfigurationOutcome.SUCCESS
        else:
            outcome = ConfigurationOutcome.OBJECTIVE_FAILURE
        progress = classify_progress(self.memory, result.tool_args, observation)
        apply_progress(self.memory, progress)

        self.memory.experiments.append(exp)
        self.graph.add_node(exp)
        # Constraint-violating runs stay on the graph but never become best_valid.
        self.memory.update_best(self.goal.primary_metric)
        record_configuration_outcome(
            self.memory,
            result.tool_args,
            experiment_id=exp_id,
            accuracy=exp.observed_result.accuracy,
            latency_ms=exp.observed_result.additional_metrics.get("latency_ms"),
            outcome=outcome,
            hypothesis=result.reasoning.hypothesis,
            failure_type=result.evaluation.failure_type,
            constraint_status=result.evaluation.constraint_status.value,
            criterion_status={k: str(v.get("status")) for k, v in status_map.items()},
        )

        # Record experiment evidence
        acc = exp.observed_result.accuracy
        latency = exp.observed_result.additional_metrics.get("latency_ms")
        if acc is not None:
            claim = f"Experiment '{exp_id}' achieved accuracy={acc:.4f}"
            if rejected:
                claim = (
                    f"Rejected experiment '{exp_id}': accuracy={acc:.4f}"
                    + (f", latency_ms={latency}" if latency is not None else "")
                    + f" ({result.evaluation.failure_type.value if result.evaluation.failure_type else 'constraint/regression'})"
                )
            self.ledger.record_experiment_result(
                memory=self.memory,
                claim=claim,
                experiment_id=exp_id,
                git_commit=exp.git_commit or "",
            )

        self.logger.log(EventType.EXPERIMENT_STARTED, experiment_id=exp_id, parent_id=parent_id, hypothesis=exp.hypothesis)
        self.logger.log(
            EventType.EXPERIMENT_COMPLETED,
            experiment_id=exp_id,
            parent_id=parent_id,
            result=exp.observed_result.model_dump(mode="json"),
            configuration=params,
            hypothesis=exp.hypothesis,
            reason=exp.reason,
            trigger=exp.trigger,
            outcome=outcome.value,
        )
        self.logger.log(
            EventType.CRITERION_EVALUATED,
            experiment_id=exp_id,
            criterion_status=status_map,
            outcome=outcome.value,
            rejected=rejected,
        )

    def _handle_failed_experiment(self, result: StepResult) -> None:
        """Record a tool/schema failure against the proposed configuration."""
        error = result.tool_response.error or ""
        schema = result.was_schema_error or "unrecognized arguments" in error.lower()
        outcome = (
            ConfigurationOutcome.INVALID_CONFIGURATION
            if schema
            else ConfigurationOutcome.TOOL_FAILURE
        )
        record_configuration_outcome(
            self.memory,
            result.tool_args,
            experiment_id=result.tool_args.get("experiment_id"),
            accuracy=None,
            latency_ms=None,
            outcome=outcome,
            hypothesis=result.reasoning.hypothesis,
            failure_type=result.evaluation.failure_type,
            constraint_status="n/a",
        )
        if schema:
            fp = configuration_fingerprint(result.tool_args)
            self.memory.rejected_configs[fp] = error or outcome.value

    def _maybe_independent_evaluation(self, result: StepResult) -> bool | None:
        """Re-score a candidate that already passed the deterministic evaluator.

        Returns True if verification agrees or is not required to close an
        optimization contract, False if metrics disagree, None if skipped.
        """
        if result.tool_name == "run_independent_evaluation":
            data = result.tool_response.data
            consistent = data.get("consistent")
            if consistent is False:
                result.evaluation.failure_type = FailureType.RESULT_INCONSISTENCY
                result.evaluation.decision = Decision.DIAGNOSE_AND_RECOVER
                self.memory.failures.append(Failure(
                    failure_type=FailureType.RESULT_INCONSISTENCY,
                    description="Independent evaluation disagreed with the candidate metrics",
                    experiment_id=result.tool_args.get("experiment_id"),
                    step_index=result.step_index,
                ))
                self.memory.reproduction.independently_verified = False
                self.memory.reproduction.verdict = ReproductionVerdict.REFUTED
                return False
            if consistent is True:
                self.memory.reproduction.independently_verified = True
                self.memory.independently_verified_experiment_id = result.tool_args.get("experiment_id")
            return True
        if result.tool_name != "run_experiment":
            return True
        if "run_independent_evaluation" not in self.registry:
            self.memory.reproduction.limitation = (
                "Independent evaluation tool is not available; metric success is not independently verified"
            )
            self.memory.reproduction.independently_verified = None
            return True
        exp_id = result.tool_args.get("experiment_id") or (
            result.tool_response.data.get("experiment_id") if result.tool_response.data else None
        )
        if not exp_id:
            return True
        if self.memory.independently_verified_experiment_id == exp_id:
            return True
        try:
            tool = self.registry.get("run_independent_evaluation")
            request_model = tool.request_model
            response = tool.execute(request_model(experiment_id=exp_id))
        except Exception as exc:
            self.memory.reproduction.limitation = f"Independent evaluation could not be run: {exc}"
            self.memory.reproduction.independently_verified = None
            self.memory.unresolved_questions.append(self.memory.reproduction.limitation)
            return True
        self.logger.log(
            EventType.TOOL_RESULT,
            tool_name="run_independent_evaluation",
            success=response.success,
            error=response.error,
            data_keys=list(response.data.keys()),
        )
        if not response.success:
            self.memory.reproduction.limitation = (
                response.error or "Independent evaluation failed; not claiming verified reproduction"
            )
            self.memory.reproduction.independently_verified = None
            self.memory.unresolved_questions.append(self.memory.reproduction.limitation)
            return True
        data = response.data
        consistent = data.get("consistent")
        if consistent is False:
            result.evaluation.failure_type = FailureType.RESULT_INCONSISTENCY
            result.evaluation.decision = Decision.DIAGNOSE_AND_RECOVER
            self.memory.failures.append(Failure(
                failure_type=FailureType.RESULT_INCONSISTENCY,
                description=(
                    f"Independent evaluation discrepancy={data.get('discrepancy')} "
                    f"eval={data.get('eval_accuracy')} training={data.get('training_accuracy')}"
                ),
                experiment_id=exp_id,
                step_index=result.step_index,
            ))
            self.memory.reproduction.independently_verified = False
            self.memory.reproduction.verdict = ReproductionVerdict.REFUTED
            return False
        self.memory.reproduction.independently_verified = True if consistent is True else None
        self.memory.independently_verified_experiment_id = exp_id
        if data.get("limitation"):
            self.memory.reproduction.limitation = str(data.get("limitation"))
        return True

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
                    h.status = HypothesisLifecycle.TESTED
                    if eval_status == "supported":
                        h.status = HypothesisLifecycle.SUPPORTED
                    elif eval_status == "rejected":
                        h.status = HypothesisLifecycle.REJECTED
                    elif eval_status == "inconclusive":
                        h.status = HypothesisLifecycle.INCONCLUSIVE
                    break

    def _update_reproduction_assessment(self, result: StepResult) -> None:
        """Persist the stricter scientific verdict separately from metric success."""
        facts = result.evaluation.goal_check or {}
        reproduction = facts.get("reproduction") if isinstance(facts, dict) else None
        if not reproduction:
            return
        self.memory.reproduction = ReproductionAssessment(
            metric_success=bool(reproduction.get("metric_success", False)),
            methodology_aligned=reproduction.get("methodology_aligned"),
            independently_verified=reproduction.get("independently_verified"),
            verdict=ReproductionVerdict(reproduction.get("verdict", "inconclusive")),
            rationale=reproduction.get("rationale", ""),
        )

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

    def _is_rejected_experiment(self, result: StepResult) -> bool:
        """True when the evaluator forbade treating this run as a legal best."""
        ev = result.evaluation
        return (
            ev.constraint_status == ConstraintStatus.VIOLATED
            or ev.failure_type == FailureType.REGRESSION
        )

    # ── Failure handling ─────────────────────────────────────────────────────

    def _handle_failure(self, result: StepResult) -> bool:
        """Dispatch a concrete recovery strategy and preserve its audit trail."""
        if not self.memory.failures or result.evaluation.failure_type is None:
            return False
        failure_type = result.evaluation.failure_type.value
        self.logger.log(EventType.RECOVERY_STARTED, failure_type=failure_type)
        # Strategy owns the policy: self-correcting recovery or naive retry.
        original_action = copy.deepcopy(result.tool_args)
        decision = self.strategy.handle_result(result, self.memory)
        self._last_strategy_decision = decision
        outcome = decision.recovery
        self.memory_store.save(self.memory)
        repaired_action = decision.repaired_tool_args
        if repaired_action:
            self.logger.log(
                EventType.ACTION_REPAIRED,
                failure_type=failure_type,
                original_action=original_action,
                error=result.tool_response.error,
                repair=decision.reason,
                repaired_action=repaired_action,
            )
        self.logger.log(
            EventType.RECOVERY_COMPLETED,
            success=decision.success,
            failure_type=failure_type,
            observed_failure=result.evaluation.rationale,
            recovery_strategy=outcome.strategy if outcome else decision.action,
            original_action=original_action,
            repaired_action=repaired_action,
            result=decision.reason,
            action=decision.reason,
            repaired_experiment_id=(
                decision.repaired_tool_args.get("experiment_id")
                if decision.repaired_tool_args
                else None
            ),
        )
        if decision.action == "recover":
            # Get the planner off self_evaluation so the next action is a new run.
            self._try_transition(AgentPhase.DIAGNOSING)
            self._try_transition(AgentPhase.RECOVERING)
            self._try_transition(AgentPhase.REPLANNING)
            self._try_transition(AgentPhase.EXPERIMENT_PLANNING)
            self.logger.log(EventType.REPLAN, plan_version=self.memory.plan_version, reason=decision.reason)
        if not decision.success:
            self.logger.log(EventType.FAILURE_DETECTED, reason=decision.reason)
        return decision.success

    def _repair_schema_args(self, tool_name: str, args: dict[str, Any], errors: list[dict[str, Any]]) -> dict[str, Any]:
        """Auto-repair common schema errors using Pydantic field error feedback."""
        repaired = canonicalize_tool_args(copy.deepcopy(args))
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
        return canonicalize_tool_args(repaired)

    # ── Utilities ────────────────────────────────────────────────────────────

    def _create_initial_plan(self) -> None:
        """Create an adaptive plan based on the goal objective."""
        obj = self.goal.objective.lower()
        # Goals about normalization, ablation, comparison -> direct execution path
        direct_experiment_keywords = (
            "normalization", "normalize", "ablation", "compare", "effect of",
            "whether", "does", "improve", "investigate whether",
            "start from the default", "default configuration",
            "existing mnist", "existing experiment", "highest possible",
        )
        needs_literature = not any(kw in obj for kw in direct_experiment_keywords)

        if needs_literature:
            steps = [
                "Search literature for relevant papers",
                "Select best paper matching goal",
                "Extract experimental methodology",
                "Run baseline experiment",
                "Compare results with reported values",
                "Adjust methodology and re-run if needed",
                "Generate verified final report",
            ]
        else:
            # For goals about testing a specific technique, go straight to experiments
            steps = [
                "Run baseline experiment without the technique",
                "Compare baseline against target threshold",
                "Apply the technique and re-run experiment",
                "Compare improved result against target threshold",
                "Generate verified final report",
            ]
            # Immediately skip to experiment planning phase
            self._try_transition(AgentPhase.EXPERIMENT_PLANNING)

        plan = Plan(
            steps=steps,
            current_step_index=0,
            version=1,
        )
        self.memory.current_plan = plan
        self.memory_store.save(self.memory)
        self.logger.log(EventType.PLAN_CREATED, steps=plan.steps, version=1)

    def _try_transition(self, to: AgentPhase) -> None:
        """Transition to a new phase only if the state machine allows it."""
        if self.state_machine.can_transition(to):
            self._transition(to)

    def _auto_advance_phase(self, result: StepResult) -> None:
        """Automatically advance plan step and agent phase based on which tool just succeeded."""
        if not result.tool_response.success:
            return

        tool = result.tool_name
        data = result.tool_response.data

        if tool == "search_literature":
            papers = data.get("papers", [])
            if papers:
                # Persist discovered papers so the planner sees them on the next step
                existing_ids = {p.get("id") for p in self.memory.discovered_papers}
                for p in papers:
                    if p.get("id") not in existing_ids:
                        self.memory.discovered_papers.append(p)
            self.memory.current_plan.advance()
            self._try_transition(AgentPhase.PAPER_SELECTION)

        elif tool == "retrieve_paper":
            self.memory.current_plan.advance()
            self._try_transition(AgentPhase.METHOD_EXTRACTION)

        elif tool == "extract_methodology":
            self.memory.current_plan.advance()
            self._try_transition(AgentPhase.EXPERIMENT_PLANNING)

        elif tool == "inspect_methodology":
            # Move forward to experiment planning so the agent runs experiments next
            self._try_transition(AgentPhase.EXPERIMENT_PLANNING)

        elif tool == "run_experiment":
            self.memory.current_plan.advance()
            self._try_transition(AgentPhase.SELF_EVALUATION)
            # The evaluator is the only component allowed to set GOAL_ACHIEVED.
            # Metric proximity with a violated constraint is goal_drift, not success.

        elif tool == "compare_results":
            # Tools return numbers and a status string; they do not close the run.
            if data.get("status") != "REPRODUCED":
                self.memory.current_plan.advance()

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

        observation = None
        if best_result:
            observation = {
                self.goal.primary_metric: best_acc,
                "latency_ms": best_result.additional_metrics.get("latency_ms"),
                "additional_metrics": best_result.additional_metrics,
            }
        # Never complete on metric proximity alone. Only the evaluator's
        # GOAL_ACHIEVED path (which already checked constraints) marks COMPLETED.
        goal_achieved = self.state_machine.current == AgentPhase.COMPLETED
        if self.memory.terminate_reason == "no_progress":
            goal_achieved = False

        report = FinalReport(
            run_id=self.run_id,
            goal=self.goal.objective,
            completed=goal_achieved,
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
