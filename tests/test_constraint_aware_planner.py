"""Constraint-aware planner context, fingerprints, and no-progress guard."""
from __future__ import annotations

from research_repro.agent.planning_context import (
    MAX_NO_PROGRESS,
    assess_proposal,
    build_failure_context,
    complexity_score,
    configuration_fingerprint,
    criteria_breakdown,
    record_blocked_proposal,
    rejected_fingerprints,
    validity_summary,
)
from research_repro.evaluation.failure_taxonomy import FailureTaxonomyClassifier
from research_repro.evaluation.goal_checker import GoalChecker
from research_repro.memory.models import (
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


def _goal() -> GoalContract:
    return GoalContract(
        objective="Improve MNIST to at least 96% accuracy with latency under 100ms",
        primary_metric="accuracy",
        target_value=0.96,
        success_threshold=0.0,
        criterion=MetricCriterion(
            metric="accuracy", direction="maximize", target=0.96, tolerance=0.0, minimum=0.96
        ),
        constraints=[Constraint(name="latency_under_100ms", description="latency_under_100ms")],
    )


def _exp(exp_id: str, acc: float, latency: float, params: dict, status: str = "negative") -> Experiment:
    exp = Experiment(
        id=exp_id,
        hypothesis="h",
        parameters=params,
        config_fingerprint=configuration_fingerprint({"parameters": params}),
        expected_result={"accuracy": 0.96, "latency_ms": 100.0},
    )
    exp.observed_result = ExperimentResult(
        accuracy=acc, additional_metrics={"latency_ms": latency}
    )
    exp.evaluation_status = EvaluationStatus(status)
    return exp


def _memory_with_drift() -> ResearchMemory:
    mem = ResearchMemory(goal=_goal(), run_id="planner_ctx")
    exp = _exp("exp_01", 0.961, 156.9, {"hidden_size": 128, "hidden_layers": 2})
    mem.experiments.append(exp)
    mem.failures.append(
        Failure(
            failure_type=FailureType.GOAL_DRIFT,
            description="accuracy on target but latency_under_100ms violated",
            experiment_id="exp_01",
        )
    )
    return mem


def test_goal_drift_identifies_satisfied_accuracy_and_violated_latency():
    mem = _memory_with_drift()
    obs = {"accuracy": 0.961, "latency_ms": 156.9, "data": {"accuracy": 0.961, "latency_ms": 156.9}}
    facts = GoalChecker().check(mem, obs, "run_experiment").to_dict()
    taxonomy = FailureTaxonomyClassifier().classify("run_experiment", obs, facts)
    assert taxonomy.failure_type == FailureType.GOAL_DRIFT
    sat, failed, violated = criteria_breakdown(mem.goal, {"accuracy": 0.961, "latency_ms": 156.9})
    assert any("accuracy" in s for s in sat)
    assert any("latency" in f for f in failed)
    assert "latency_under_100ms" in violated
    ctx = build_failure_context(mem)
    assert ctx is not None
    assert ctx.failure_type == FailureType.GOAL_DRIFT
    assert any("accuracy" in s for s in ctx.satisfied_criteria)
    assert "latency_under_100ms" in ctx.violated_constraints
    assert ctx.observed_metrics["accuracy"] == 0.961
    assert ctx.observed_metrics["latency_ms"] == 156.9
    assert ctx.previous_configuration["hidden_size"] == 128
    assert ctx.previous_configuration["hidden_layers"] == 2
    assert ctx.required_change and "complexity" in ctx.required_change


def test_empty_parameters_fingerprint_matches_128x2_default():
    default_fp = configuration_fingerprint({"parameters": {}})
    explicit_fp = configuration_fingerprint(
        {"parameters": {"hidden_size": 128, "hidden_layers": 2, "solver": "adam", "max_iter": 100}}
    )
    assert default_fp == explicit_fp


def test_rejected_configuration_is_marked_and_blocked():
    mem = _memory_with_drift()
    fps = rejected_fingerprints(mem)
    rejected_fp = configuration_fingerprint({"parameters": {"hidden_size": 128, "hidden_layers": 2}})
    assert rejected_fp in fps
    same = assess_proposal(
        mem, "run_experiment", {"experiment_id": "exp_02", "parameters": {}}
    )
    assert same.blocked is True
    assert "NO_PROGRESS" in same.reason


def test_latency_drift_requires_complexity_decrease():
    mem = _memory_with_drift()
    prev = complexity_score({"hidden_size": 128, "hidden_layers": 2})
    repeat = assess_proposal(
        mem,
        "run_experiment",
        {"experiment_id": "exp_02", "parameters": {"hidden_size": 128, "hidden_layers": 2, "max_iter": 200}},
    )
    assert repeat.blocked is True
    smaller = assess_proposal(
        mem,
        "run_experiment",
        {"experiment_id": "exp_02", "parameters": {"hidden_size": 64, "hidden_layers": 1, "normalize": True}},
    )
    assert smaller.blocked is False
    assert complexity_score({"hidden_size": 64, "hidden_layers": 1}) < prev


def test_preserve_satisfied_accuracy_in_failure_context():
    ctx = build_failure_context(_memory_with_drift())
    assert ctx is not None
    assert any("accuracy" in s for s in ctx.satisfied_criteria)
    assert ctx.required_change
    assert "accuracy" not in ctx.required_change.lower() or "preserve" in (ctx.required_change.lower() + " ".join(ctx.satisfied_criteria).lower())
    assert "complexity" in ctx.required_change.lower() or "hidden" in str(ctx.intervention).lower()


def test_regression_context_includes_best_valid_and_lineage():
    mem = ResearchMemory(goal=_goal(), run_id="reg")
    valid = _exp("exp_04", 0.932, 76.1, {"hidden_size": 64, "hidden_layers": 1}, status="positive")
    worse = _exp("exp_05", 0.90, 70.0, {"hidden_size": 32, "hidden_layers": 1}, status="negative")
    worse.parent_id = "exp_04"
    mem.experiments.extend([valid, worse])
    mem.best_experiment_id = "exp_04"
    mem.failures.append(
        Failure(failure_type=FailureType.REGRESSION, description="worse than best valid", experiment_id="exp_05")
    )
    ctx = build_failure_context(mem)
    assert ctx is not None
    assert ctx.failure_type == FailureType.REGRESSION
    assert ctx.required_change and "best_valid" in ctx.required_change
    from research_repro.agent.planning_context import best_valid_experiment
    best = best_valid_experiment(mem)
    assert best is not None
    assert best["experiment_id"] == "exp_04"
    assert worse.parent_id == "exp_04"
    rows = validity_summary(mem)
    assert any(r["experiment_id"] == "exp_04" for r in rows)


def test_planner_coerces_bare_tool_arguments():
    from research_repro.agent.planner import Planner
    raw = Planner._coerce_planner_output({
        "experiment_id": "exp_02",
        "parameters": {"hidden_size": 64, "hidden_layers": 1},
        "timeout_seconds": 600,
    })
    assert raw["selected_tool"] == "run_experiment"
    assert raw["tool_arguments"]["parameters"]["hidden_size"] == 64
    assert "hypothesis" in raw["reasoning"]


def test_failure_context_prefers_latest_constraint_violation_over_stale_crash():
    mem = _memory_with_drift()
    mem.failures.append(Failure(failure_type=FailureType.TOOL_CRASH, description="timeout", experiment_id="gone"))
    ctx = build_failure_context(mem)
    assert ctx is not None
    assert ctx.failure_type == FailureType.GOAL_DRIFT
    assert "latency_under_100ms" in ctx.violated_constraints


def test_no_progress_terminates_after_repeated_rejected_configs():
    mem = _memory_with_drift()
    args = {"experiment_id": "exp_99", "parameters": {"hidden_size": 128, "hidden_layers": 2}}
    terminal = False
    for _ in range(MAX_NO_PROGRESS):
        assessment = assess_proposal(mem, "run_experiment", args)
        assert assessment.blocked is True
        record_blocked_proposal(mem, assessment, args)
        terminal = assessment.terminal
    assert terminal is True
    assert mem.no_progress_count >= MAX_NO_PROGRESS
    assert mem.unresolved_questions
