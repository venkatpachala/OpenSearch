"""Focused tests for the evidence-driven planner/recovery specification."""
from __future__ import annotations

from pathlib import Path

from research_repro.agent.loop import AgentLoop
from research_repro.agent.planner import Planner, PlannerOutput, ReasoningTrace
from research_repro.agent.planning_context import (
    MAX_NO_PROGRESS,
    assess_proposal,
    best_valid_experiment,
    build_failure_context,
    build_planning_evidence,
    canonicalize_parameters,
    canonicalize_tool_args,
    configuration_fingerprint,
    deterministic_next_parameters,
    ensure_falsifiable_hypothesis,
    is_generic_hypothesis,
    record_blocked_proposal,
    repair_from_argparse_error,
    validate_configuration,
)
from research_repro.agent.react import StepResult
from research_repro.config import Config
from research_repro.evaluation.evaluator import (
    ConstraintStatus,
    Decision,
    EvaluationResult,
    GoalProgress,
    HypothesisStatus,
    SelfEvaluator,
)
from research_repro.evaluation.failure_taxonomy import FailureTaxonomyClassifier
from research_repro.evaluation.goal_checker import GoalChecker
from research_repro.experiments.runner import ExperimentConfig
from research_repro.memory.models import (
    ConfigurationOutcome,
    ConfigurationRecord,
    Constraint,
    EvaluationStatus,
    Experiment,
    ExperimentResult,
    Failure,
    FailureType,
    GoalContract,
    MetricCriterion,
    ResearchMemory,
)
from research_repro.recovery.base import RecoveryContext
from research_repro.recovery.strategies import SchemaRepairStrategy
from research_repro.tools.base import Tool, ToolRegistry, ToolRequest, ToolResponse
from research_repro.tools.stub import build_stub_registry


def _goal() -> GoalContract:
    return GoalContract(
        objective=(
            "Find a configuration for the existing MNIST image-classification "
            "experiment that achieves the highest possible test accuracy while "
            "keeping latency below 100 ms."
        ),
        primary_metric="accuracy",
        target_value=0.95,
        success_threshold=0.0,
        criterion=MetricCriterion(
            metric="accuracy", direction="maximize", target=0.95, tolerance=0.0, minimum=0.95
        ),
        constraints=[Constraint(name="latency_under_100ms", description="latency_under_100ms")],
        max_experiments=6,
    )


def _exp(exp_id: str, acc: float, latency: float, params: dict, status: str = "negative") -> Experiment:
    exp = Experiment(
        id=exp_id,
        hypothesis="h",
        parameters=params,
        config_fingerprint=configuration_fingerprint({"parameters": params}),
        expected_result={"accuracy": 0.95, "latency_ms": 100.0},
    )
    exp.observed_result = ExperimentResult(accuracy=acc, additional_metrics={"latency_ms": latency})
    exp.evaluation_status = EvaluationStatus(status)
    return exp


def _eval(**kwargs) -> EvaluationResult:
    defaults = dict(
        execution_success=True,
        output_valid=True,
        metric_valid=True,
        goal_progress=GoalProgress.NEGATIVE,
        constraint_status=ConstraintStatus.ALL_MET,
        hypothesis_status=HypothesisStatus.REJECTED,
        failure_type=None,
        decision=Decision.CONTINUE,
        confidence=0.9,
        rationale="test",
    )
    defaults.update(kwargs)
    return EvaluationResult(**defaults)


# ---------------------------------------------------------------------------
# Fingerprints / meaningful change
# ---------------------------------------------------------------------------


def test_empty_and_alias_fingerprints_match_real_defaults():
    empty = configuration_fingerprint({"parameters": {}})
    explicit = configuration_fingerprint(
        {
            "parameters": {
                "hidden_size": 128,
                "hidden_layers": 2,
                "max_iter": 100,
                "lr": 0.001,
                "batch_size": 200,
                "normalize": False,
                "pca_components": 0,
                "solver": "adam",
            }
        }
    )
    aliased = configuration_fingerprint({"parameters": {"learning_rate": 0.001, "n_layers": 2}})
    assert empty == explicit == aliased
    assert "batch_size=200" in empty


def test_experiment_id_is_not_a_meaningful_change():
    a = configuration_fingerprint({"experiment_id": "exp_01", "parameters": {"hidden_size": 64}})
    b = configuration_fingerprint({"experiment_id": "exp_99", "parameters": {"hidden_size": 64}})
    assert a == b


def test_unknown_and_out_of_range_configs_are_invalid():
    assert validate_configuration({"dropout": 0.5})
    assert validate_configuration({"hidden_size": 0})
    assert validate_configuration({"solver": "rmsprop"})
    assert not validate_configuration({"hidden_size": 64, "lr": 0.001, "normalize": True})


# ---------------------------------------------------------------------------
# Duplicate tested configuration
# ---------------------------------------------------------------------------


def test_duplicate_tested_configuration_is_blocked_without_execution():
    mem = ResearchMemory(goal=_goal(), run_id="dup")
    exp = _exp("exp_a", 0.9125, 76.1, {"hidden_size": 64, "hidden_layers": 1}, status="negative")
    mem.experiments.append(exp)
    mem.tested_configs[exp.config_fingerprint] = ConfigurationRecord(
        fingerprint=exp.config_fingerprint,
        config={"hidden_size": 64, "hidden_layers": 1},
        experiment_id="exp_a",
        accuracy=0.9125,
        latency_ms=76.1,
        outcome=ConfigurationOutcome.OBJECTIVE_FAILURE,
    )
    assessment = assess_proposal(
        mem,
        "run_experiment",
        {"experiment_id": "exp_b", "parameters": {"hidden_size": 64, "hidden_layers": 1}},
    )
    assert assessment.blocked is True
    assert assessment.block_code == "duplicate_tested_configuration"
    assert assessment.previous_experiment == "exp_a"
    assert assessment.previous_outcome == "OBJECTIVE_FAILURE"


# ---------------------------------------------------------------------------
# Constraint-invalid best-valid semantics
# ---------------------------------------------------------------------------


def test_constraint_invalid_high_accuracy_is_not_best_valid(tmp_path):
    loop = AgentLoop(_goal(), "best_valid", Config(runs_dir=tmp_path), build_stub_registry())
    drifted = StepResult(
        step_index=1,
        reasoning=ReasoningTrace(goal_relevance="g", evidence_basis="e", hypothesis="baseline", intended_action="run"),
        tool_name="run_experiment",
        tool_args={"experiment_id": "exp_01", "parameters": {}},
        tool_response=ToolResponse.ok(experiment_id="exp_01", accuracy=0.961, latency_ms=156.9, f1=0.96, loss=0.1),
        evaluation=_eval(
            constraint_status=ConstraintStatus.VIOLATED,
            failure_type=FailureType.GOAL_DRIFT,
            decision=Decision.DIAGNOSE_AND_RECOVER,
            rationale="accuracy on target but latency violated",
        ),
        decision=Decision.DIAGNOSE_AND_RECOVER,
    )
    legal = StepResult(
        step_index=2,
        reasoning=ReasoningTrace(goal_relevance="g", evidence_basis="e", hypothesis="smaller", intended_action="run"),
        tool_name="run_experiment",
        tool_args={"experiment_id": "exp_02", "parameters": {"hidden_size": 64, "hidden_layers": 1}},
        tool_response=ToolResponse.ok(experiment_id="exp_02", accuracy=0.9185, latency_ms=76.1, f1=0.91, loss=0.2),
        evaluation=_eval(decision=Decision.CONTINUE, goal_progress=GoalProgress.NEGATIVE),
        decision=Decision.CONTINUE,
    )
    loop._handle_experiment_result(drifted)
    loop._handle_experiment_result(legal)
    assert loop.memory.best_experiment_id == "exp_02"
    assert loop.memory.best_observed_accuracy_id == "exp_01"
    best = best_valid_experiment(loop.memory)
    assert best is not None
    assert best["experiment_id"] == "exp_02"
    assert loop.memory.tested_configs[configuration_fingerprint({"parameters": {}})].outcome == ConfigurationOutcome.CONSTRAINT_VIOLATION


def test_legal_accuracy_miss_is_not_goal_drift_and_does_not_recover():
    from research_repro.evaluation.evaluator import SelfEvaluator
    mem = ResearchMemory(goal=_goal(), run_id="acc_miss")
    evaluator = SelfEvaluator()
    result = evaluator.evaluate(
        mem,
        "run_experiment",
        {"experiment_id": "exp_03", "parameters": {"hidden_size": 64, "hidden_layers": 1}},
        {"success": True, "data": {"accuracy": 0.932, "latency_ms": 76.1}},
        "h",
    )
    assert result.constraint_status == ConstraintStatus.ALL_MET
    assert result.failure_type is None
    assert result.decision == Decision.CONTINUE
    assert result.goal_progress == GoalProgress.NEGATIVE


def test_goal_completion_is_and_of_all_criteria():
    goal = _goal()
    assert not goal.meets_contract(0.961, {"latency_ms": 156.9})
    assert not goal.meets_contract(0.9125, {"latency_ms": 76.1})
    assert goal.meets_contract(0.956, {"latency_ms": 80.0})
    facts_fail_lat = GoalChecker().check(
        ResearchMemory(goal=goal, run_id="and1"),
        {"accuracy": 0.961, "latency_ms": 156.9, "data": {"accuracy": 0.961, "latency_ms": 156.9}},
        "run_experiment",
    )
    assert facts_fail_lat.metric.achieved is True
    assert facts_fail_lat.any_constraint_violated is True
    assert facts_fail_lat.all_good is False
    facts_fail_acc = GoalChecker().check(
        ResearchMemory(goal=goal, run_id="and2"),
        {"accuracy": 0.9125, "latency_ms": 76.1, "data": {"accuracy": 0.9125, "latency_ms": 76.1}},
        "run_experiment",
    )
    assert facts_fail_acc.metric.achieved is False
    assert facts_fail_acc.any_constraint_violated is False
    assert facts_fail_acc.all_good is False


# ---------------------------------------------------------------------------
# Accuracy-blocking planner evidence
# ---------------------------------------------------------------------------


def test_accuracy_failure_context_preserves_latency_and_asks_for_accuracy():
    mem = ResearchMemory(goal=_goal(), run_id="acc_block")
    mem.experiments.append(
        _exp("exp_02", 0.9125, 76.1, {"hidden_size": 64, "hidden_layers": 1}, status="negative")
    )
    mem.best_experiment_id = "exp_02"
    evidence = build_planning_evidence(mem)
    assert evidence.blocking_criterion == "accuracy"
    assert "accuracy is currently the blocking criterion" in evidence.guidance.lower()
    assert "preserving latency" in evidence.guidance.lower()
    assert "improve predictive performance" in evidence.required_change
    next_proposal = assess_proposal(
        mem,
        "run_experiment",
        {"experiment_id": "exp_03", "parameters": {"hidden_size": 64, "hidden_layers": 1, "normalize": True}},
    )
    assert next_proposal.blocked is False
    assert next_proposal.fingerprint != configuration_fingerprint(
        {"parameters": {"hidden_size": 64, "hidden_layers": 1}}
    )


# ---------------------------------------------------------------------------
# Schema recovery
# ---------------------------------------------------------------------------


def test_learning_rate_alias_is_repaired_to_lr():
    original = {"experiment_id": "exp_01", "parameters": {"learning_rate": 0.001, "hidden_size": 64}}
    repaired = repair_from_argparse_error(original, "unrecognized arguments: --learning-rate 0.001")
    assert repaired["parameters"]["lr"] == 0.001
    assert "learning_rate" not in repaired["parameters"]
    cfg = ExperimentConfig(
        experiment_id="exp_01",
        script_path=Path("environments/image_classification/train.py"),
        output_dir=Path("runs/tmp"),
        parameters={"learning_rate": 0.001, "hidden_size": 64},
    )
    args = cfg.to_args()
    assert "--lr" in args
    assert "--learning-rate" not in args


def test_schema_recovery_does_not_repeat_invalid_action():
    mem = ResearchMemory(goal=_goal(), run_id="schema")
    ctx = RecoveryContext(
        memory=mem,
        failure=Failure(failure_type=FailureType.TOOL_SCHEMA_ERROR, description="bad flag"),
        tool_args={"experiment_id": "exp_01", "parameters": {"learning_rate": 0.001, "hidden_size": 64}},
        tool_response={"success": False, "error": "unrecognized arguments: --learning-rate 0.001"},
    )
    outcome = SchemaRepairStrategy().recover(ctx)
    assert outcome.success is True
    assert outcome.retry is True
    assert outcome.updates["tool_args"]["parameters"]["lr"] == 0.001
    assert outcome.updates["tool_args"]["experiment_id"] == "exp_01"


def test_schema_recovery_exhausts_honestly_when_unrepairable():
    mem = ResearchMemory(goal=_goal(), run_id="schema_ex")
    ctx = RecoveryContext(
        memory=mem,
        failure=Failure(failure_type=FailureType.TOOL_SCHEMA_ERROR, description="unknown"),
        tool_args={"experiment_id": "exp_01", "parameters": {"not_a_real_flag": 1}},
        tool_response={"success": False, "error": "unrecognized arguments: --not-a-real-flag 1"},
    )
    outcome = SchemaRepairStrategy().recover(ctx)
    assert outcome.success is False
    assert "exhausted" in (outcome.unresolved or outcome.action_taken).lower()


def test_argparse_error_is_classified_as_schema_not_crash():
    taxonomy = FailureTaxonomyClassifier().classify(
        "run_experiment",
        {"success": False, "error": "Experiment failed (exit code 2). stderr: unrecognized arguments: --learning-rate 0.001"},
        {},
    )
    assert taxonomy.failure_type == FailureType.TOOL_SCHEMA_ERROR


# ---------------------------------------------------------------------------
# No-progress including inner replans / hypotheses
# ---------------------------------------------------------------------------


def test_repeated_duplicates_increment_no_progress_to_termination():
    mem = ResearchMemory(goal=_goal(), run_id="np")
    exp = _exp("exp_01", 0.961, 156.9, {"hidden_size": 128, "hidden_layers": 2})
    mem.experiments.append(exp)
    mem.failures.append(Failure(failure_type=FailureType.GOAL_DRIFT, description="latency", experiment_id="exp_01"))
    args = {"experiment_id": "exp_99", "parameters": {}}
    terminal = False
    for _ in range(MAX_NO_PROGRESS):
        assessment = assess_proposal(mem, "run_experiment", args)
        assert assessment.blocked is True
        record_blocked_proposal(mem, assessment, args)
        terminal = assessment.terminal
    assert terminal is True
    assert mem.no_progress_count >= MAX_NO_PROGRESS
    assert mem.terminate_reason == "no_progress"


def test_generic_hypothesis_is_rejected_for_evidence_based_fallback():
    assert is_generic_hypothesis("The proposed experiment configuration should be evaluated.")
    assert is_generic_hypothesis("This configuration may work.")
    mem = ResearchMemory(goal=_goal(), run_id="hyp")
    mem.experiments.append(_exp("exp_02", 0.9125, 76.1, {"hidden_size": 64, "hidden_layers": 1}))
    text = ensure_falsifiable_hypothesis(
        mem,
        "The proposed experiment configuration should be evaluated.",
        {"hidden_size": 64, "hidden_layers": 1, "normalize": True},
    )
    assert not is_generic_hypothesis(text)
    assert "0.9125" in text
    assert "100" in text
    assert "normalize" in text.lower() or "->" in text


def test_planner_finalize_rewrites_generic_hypothesis():
    mem = ResearchMemory(goal=_goal(), run_id="fin")
    mem.experiments.append(_exp("exp_02", 0.9125, 76.1, {"hidden_size": 64, "hidden_layers": 1}))
    planner = Planner()
    raw = Planner._coerce_planner_output({
        "experiment_id": "exp_03",
        "parameters": {"hidden_size": 64, "hidden_layers": 1, "normalize": True},
    })
    output = planner._finalize_output(mem, PlannerOutput(**raw))
    assert not is_generic_hypothesis(output.reasoning.hypothesis)
    assert output.tool_arguments["parameters"].get("normalize") is True


# ---------------------------------------------------------------------------
# Independent evaluation does not add noise
# ---------------------------------------------------------------------------


def test_independent_eval_script_rescores_artifacts(tmp_path):
    import json
    import subprocess
    import sys

    script = Path(__file__).resolve().parents[1] / "environments" / "image_classification" / "train.py"
    eval_script = Path(__file__).resolve().parents[1] / "environments" / "image_classification" / "evaluate.py"
    out = tmp_path / "exp_syn"
    proc = subprocess.run(
        [sys.executable, str(script), "--output-dir", str(out), "--synthetic", "--max-iter", "20", "--hidden-size", "32", "--hidden-layers", "1"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "model.joblib").exists()
    assert (out / "eval_split.npz").exists()
    eval_out = out / "eval"
    eval_proc = subprocess.run(
        [sys.executable, str(eval_script), "--model-dir", str(out), "--output-dir", str(eval_out)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert eval_proc.returncode == 0, eval_proc.stderr
    metrics = json.loads((out / "metrics.json").read_text())
    eval_metrics = json.loads((eval_out / "eval_metrics.json").read_text())
    assert eval_metrics["independently_verified"] is True
    assert eval_metrics["consistent"] is True
    assert abs(eval_metrics["eval_accuracy"] - metrics["accuracy"]) < 0.01


def test_train_py_has_no_silent_synthetic_fallback():
    src = (Path(__file__).resolve().parents[1] / "environments" / "image_classification" / "train.py").read_text(encoding="utf-8")
    assert "--synthetic" in src
    assert "refusing to substitute synthetic data" in src.lower() or "mnist unavailable" in src.lower()
    eval_src = (Path(__file__).resolve().parents[1] / "environments" / "image_classification" / "evaluate.py").read_text(encoding="utf-8")
    assert "RandomState" not in eval_src
    assert "normal(0" not in eval_src


# ---------------------------------------------------------------------------
# Integration: scripted MNIST self-correction sequence
# ---------------------------------------------------------------------------


class ScriptedMNISTRequest(ToolRequest):
    experiment_id: str
    parameters: dict = {}
    timeout_seconds: int = 600
    seed: int = 42
    parent_experiment_id: str | None = None


class ScriptedMNISTTool(Tool):
    name = "run_experiment"
    description = "Deterministic MNIST stand-in for the specification sequence."
    request_model = ScriptedMNISTRequest
    calls: list[dict] = []

    def execute(self, request: ScriptedMNISTRequest) -> ToolResponse:
        params = canonicalize_parameters(request.parameters or {})
        hidden = int(params.get("hidden_size", 128))
        layers = int(params.get("hidden_layers", 2))
        normalize = bool(params.get("normalize", False))
        self.calls.append({"experiment_id": request.experiment_id, "parameters": params})
        if hidden * layers >= 256:
            acc, lat = 0.961, 156.9
        elif not normalize:
            acc, lat = 0.9125, 76.1
        else:
            acc, lat = 0.956, 80.0
        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            accuracy=acc,
            latency_ms=lat,
            f1=acc,
            loss=0.1,
            runtime_seconds=1.0,
        )


class ScriptedPlanner:
    def plan(self, memory, registry, rejection_hint=None):
        n = len(memory.experiments)
        if n == 0:
            params = {}
            hyp = "Baseline default configuration should be measured against the contract."
            exp_id = "exp_01"
        elif any("latency" in f.lower() for f in build_planning_evidence(memory).failed_criteria):
            params = {"hidden_size": 64, "hidden_layers": 1}
            hyp = (
                "Given accuracy=0.961 with latency=156.9ms above the 100ms cap, reducing "
                "hidden_size 128->64 and hidden_layers 2->1 should cut complexity enough "
                "to satisfy latency while preserving as much accuracy as possible."
            )
            exp_id = "exp_02"
        else:
            params = {"hidden_size": 64, "hidden_layers": 1, "normalize": True}
            hyp = (
                "Given the previous 64x1 run observed accuracy=0.9125 below 0.95 with legal "
                "latency=76.1ms, enabling normalization may recover accuracy while remaining "
                "under the latency cap of 100ms."
            )
            exp_id = "exp_03"
        if rejection_hint:
            params = {**params, "max_iter": 80, "normalize": True, "hidden_size": 48, "hidden_layers": 1}
            hyp = ensure_falsifiable_hypothesis(memory, hyp, params)
            exp_id = f"exp_repair_{memory.no_progress_count}"
        return PlannerOutput(
            reasoning=ReasoningTrace(
                goal_relevance="Advance the MNIST accuracy/latency contract.",
                evidence_basis=build_planning_evidence(memory).guidance,
                hypothesis=hyp,
                intended_action=f"run_experiment {params}",
            ),
            selected_tool="run_experiment",
            tool_arguments={"experiment_id": exp_id, "parameters": params, "timeout_seconds": 30},
            confidence=0.8,
        )


def test_deterministic_fallback_reduces_complexity_after_latency_drift():
    mem = ResearchMemory(goal=_goal(), run_id="fb")
    mem.experiments.append(_exp("exp_01", 0.961, 156.9, {"hidden_size": 128, "hidden_layers": 2}))
    mem.failures.append(Failure(failure_type=FailureType.GOAL_DRIFT, description="latency", experiment_id="exp_01"))
    nxt = deterministic_next_parameters(mem)
    assert nxt is not None
    from research_repro.agent.planning_context import complexity_score
    assert complexity_score(nxt) < complexity_score({"hidden_size": 128, "hidden_layers": 2})
    assert not assess_proposal(mem, "run_experiment", {"experiment_id": "exp_02", "parameters": nxt}).blocked


def test_planner_replaces_blocked_default_with_fallback():
    mem = ResearchMemory(goal=_goal(), run_id="fb2")
    mem.experiments.append(_exp("exp_01", 0.961, 156.9, {"hidden_size": 128, "hidden_layers": 2}))
    mem.tested_configs[configuration_fingerprint({"parameters": {}})] = ConfigurationRecord(
        fingerprint=configuration_fingerprint({"parameters": {}}),
        config={"hidden_size": 128, "hidden_layers": 2},
        experiment_id="exp_01",
        accuracy=0.961,
        latency_ms=156.9,
        outcome=ConfigurationOutcome.CONSTRAINT_VIOLATION,
    )
    mem.failures.append(Failure(failure_type=FailureType.GOAL_DRIFT, description="latency", experiment_id="exp_01"))
    planner = Planner()
    raw = Planner._coerce_planner_output({"parameters": {}})
    output = planner._finalize_output(mem, PlannerOutput(**raw))
    assert configuration_fingerprint(output.tool_arguments) != configuration_fingerprint({"parameters": {}})
    assert not assess_proposal(mem, "run_experiment", output.tool_arguments).blocked


def test_optimization_goal_skips_literature_and_starts_at_experiments(tmp_path):
    loop = AgentLoop(_goal(), "direct", Config(runs_dir=tmp_path), build_stub_registry())
    loop._create_initial_plan()
    assert loop.memory.current_phase == "experiment_planning"
    assert any("baseline" in s.lower() or "experiment" in s.lower() for s in loop.memory.current_plan.steps)


def test_scripted_self_correction_sequence(tmp_path):
    ScriptedMNISTTool.calls = []
    registry = ToolRegistry()
    tool = ScriptedMNISTTool()
    registry.register(tool)
    from research_repro.tools.stub import StubIndependentEvalTool
    registry.register(StubIndependentEvalTool())

    loop = AgentLoop(_goal(), "seq", Config(runs_dir=tmp_path), registry)
    loop.planner = ScriptedPlanner()
    loop.react.planner = loop.planner
    loop.evaluator._llm_evaluate = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no llm"))

    report = loop.run()
    fps = [configuration_fingerprint({"parameters": c["parameters"]}) for c in tool.calls]
    assert len(fps) == len(set(fps)), "duplicate configuration was executed"
    assert len(loop.memory.experiments) >= 2
    first = loop.memory.experiments[0]
    assert first.observed_result.accuracy == 0.961
    assert first.observed_result.additional_metrics["latency_ms"] == 156.9
    assert first.parent_id is None
    assert loop.memory.best_experiment_id != first.id
    assert any(e.parent_id == first.id for e in loop.memory.experiments[1:])
    evidence_after_small = None
    for exp in loop.memory.experiments:
        if exp.observed_result and exp.observed_result.additional_metrics.get("latency_ms") == 76.1:
            evidence_after_small = exp
            break
    assert evidence_after_small is not None
    ctx = build_planning_evidence(loop.memory)
    assert any("normalize" in (e.parameters or {}) and e.parameters.get("normalize") for e in loop.memory.experiments) or any(
        c["parameters"].get("normalize") for c in tool.calls
    )
    for hyp in loop.memory.hypotheses:
        assert not is_generic_hypothesis(hyp.statement)
    if report.completed:
        best = loop.memory.get_best_result()
        assert best is not None
        assert best.accuracy >= 0.95
        assert best.additional_metrics.get("latency_ms") <= 100
    else:
        # Budget or independent-eval limitation may stop short; still must not
        # declare the illegal baseline as success.
        assert loop.state_machine.current.value != "completed" or loop.memory.best_experiment_id != first.id
    events = (tmp_path / "seq" / "run.jsonl").read_text(encoding="utf-8")
    assert "proposal_created" in events or "PROPOSAL_CREATED" in events or '"proposal_created"' in events
    assert "goal_drift" in events or "GOAL_DRIFT" in events or "goal_drift" in events.lower()
