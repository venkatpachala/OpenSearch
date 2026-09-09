"""Structured planner context: failures, fingerprints, and proposal guards.

The evaluator remains authoritative. This module only turns evaluator facts
into actionable planner input and blocks silent repeats of rejected configs.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ..memory.models import Experiment, FailureType, GoalContract, ResearchMemory


# Defaults of environments/image_classification/train.py so {} fingerprints
# as the same 128x2 Adam run the sandbox actually executes.
DEFAULT_TRAIN_PARAMS: dict[str, Any] = {
    "hidden_size": 128,
    "hidden_layers": 2,
    "max_iter": 100,
    "lr": 0.001,
    "batch_size": 200,
    "normalize": False,
    "pca_components": 0,
    "solver": "adam",
    "seed": 42,
}

FINGERPRINT_KEYS = (
    "solver",
    "lr",
    "hidden_size",
    "hidden_layers",
    "normalize",
    "pca_components",
    "max_iter",
)

# Domain knowledge: which knobs influence which constraint.
# Not a prescribed configuration — the planner still chooses values.
CONSTRAINT_INTERVENTIONS: dict[str, dict[str, Any]] = {
    "latency_under_100ms": {
        "reduce": ["hidden_size", "hidden_layers"],
        "consider": ["pca_components"],
        "preserve": ["normalize"],
        "required_change": "reduce model complexity (hidden_size and/or hidden_layers)",
    },
}

LATENCY_LIMIT_MS = 100.0
MAX_NO_PROGRESS = 3


class PlannerFailureContext(BaseModel):
    failure_type: FailureType
    experiment_id: str | None = None
    violated_constraints: list[str] = Field(default_factory=list)
    satisfied_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    observed_metrics: dict[str, float] = Field(default_factory=dict)
    target_metrics: dict[str, float] = Field(default_factory=dict)
    previous_configuration: dict[str, Any] = Field(default_factory=dict)
    rejected_configurations: list[dict[str, Any]] = Field(default_factory=list)
    diagnosis: str | None = None
    evidence: list[str] = Field(default_factory=list)
    required_change: str | None = None
    intervention: dict[str, Any] | None = None


class ProposalAssessment(BaseModel):
    blocked: bool = False
    terminal: bool = False
    reason: str = ""
    fingerprint: str = ""


def experiment_parameters(args: dict[str, Any] | None) -> dict[str, Any]:
    if not args:
        return {}
    params = args.get("parameters") if isinstance(args.get("parameters"), dict) else args
    return dict(params) if isinstance(params, dict) else {}


def normalize_parameters(params: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_TRAIN_PARAMS)
    if params:
        for key, value in params.items():
            if value is not None:
                merged[key] = value
    return merged


def configuration_fingerprint(args: dict[str, Any] | None) -> str:
    norm = normalize_parameters(experiment_parameters(args))
    return "|".join(f"{key}={norm.get(key)}" for key in FINGERPRINT_KEYS)


def complexity_score(params: dict[str, Any] | None) -> int:
    norm = normalize_parameters(experiment_parameters(params) or params)
    try:
        return int(norm.get("hidden_size", 128)) * int(norm.get("hidden_layers", 2))
    except (TypeError, ValueError):
        return 256


def _observation_from_experiment(exp: Experiment) -> dict[str, Any]:
    result = exp.observed_result
    if result is None:
        return {}
    obs: dict[str, Any] = {
        "accuracy": result.accuracy,
        "f1": result.f1,
        "loss": result.loss,
        "runtime_seconds": result.runtime_seconds,
        **result.additional_metrics,
        "additional_metrics": result.additional_metrics,
    }
    return {k: v for k, v in obs.items() if v is not None}


def criteria_breakdown(goal: GoalContract, observation: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    satisfied: list[str] = []
    failed: list[str] = []
    violated: list[str] = []
    metric = goal.primary_metric
    observed = observation.get(metric)
    if observed is None and metric == "accuracy":
        observed = observation.get("accuracy")
    if isinstance(observed, (int, float)):
        label = _metric_criterion_label(goal)
        if goal.is_achieved(float(observed)):
            satisfied.append(label)
        else:
            failed.append(label)
    for constraint in goal.constraints:
        name = constraint.name
        if "latency" in name.lower():
            latency = observation.get("latency_ms")
            label = f"latency < {int(LATENCY_LIMIT_MS)}ms"
            if latency is None:
                continue
            if float(latency) <= LATENCY_LIMIT_MS:
                satisfied.append(label)
            else:
                failed.append(label)
                violated.append(name)
        elif not goal.constraints_hold(observation) and name not in violated:
            # Non-latency named constraints that fail constraints_hold.
            if "latency" not in name.lower():
                failed.append(name)
                violated.append(name)
    return satisfied, failed, violated


def _metric_criterion_label(goal: GoalContract) -> str:
    if goal.criterion and goal.criterion.direction == "maximize":
        minimum = goal.criterion.minimum
        if minimum is None and goal.criterion.target is not None:
            minimum = goal.criterion.target - (goal.criterion.tolerance or 0.0)
        if minimum is not None:
            return f"{goal.primary_metric} >= {minimum}"
    if goal.target_value is not None:
        return f"{goal.primary_metric} within {goal.success_threshold} of {goal.target_value}"
    return goal.primary_metric


def experiment_is_rejected(memory: ResearchMemory, exp: Experiment) -> bool:
    if exp.evaluation_status.value == "negative" and not memory.goal.constraints_hold(_observation_from_experiment(exp)):
        return True
    for failure in memory.failures:
        if failure.experiment_id == exp.id and failure.failure_type in (
            FailureType.GOAL_DRIFT,
            FailureType.REGRESSION,
            FailureType.RESULT_INCONSISTENCY,
        ):
            return True
    return False


def rejected_fingerprints(memory: ResearchMemory) -> set[str]:
    out: set[str] = set()
    for exp in memory.experiments:
        if experiment_is_rejected(memory, exp):
            out.add(exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters}))
    for blocked in memory.blocked_proposals:
        fp = blocked.get("fingerprint")
        if fp:
            out.add(str(fp))
    return out


def rejected_configuration_rows(memory: ResearchMemory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in memory.experiments:
        if not experiment_is_rejected(memory, exp):
            continue
        obs = _observation_from_experiment(exp)
        _sat, failed, violated = criteria_breakdown(memory.goal, obs)
        rows.append({
            "experiment_id": exp.id,
            "fingerprint": exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters}),
            "configuration": normalize_parameters(exp.parameters),
            "observed_metrics": {k: v for k, v in obs.items() if isinstance(v, (int, float))},
            "reason": ", ".join(violated or failed) or "rejected",
        })
    for blocked in memory.blocked_proposals:
        rows.append(blocked)
    return rows


def validity_summary(memory: ResearchMemory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for exp in memory.experiments:
        if not exp.observed_result:
            continue
        obs = _observation_from_experiment(exp)
        sat, failed, _violated = criteria_breakdown(memory.goal, obs)
        acc = obs.get("accuracy")
        lat = obs.get("latency_ms")
        if not failed:
            status = "VALID"
        elif any("latency" in f for f in failed) and not memory.goal.constraints_hold(obs):
            status = "INVALID: latency"
        else:
            status = "INVALID: " + ", ".join(failed)
        rows.append({
            "experiment_id": exp.id,
            "accuracy": acc,
            "latency_ms": lat,
            "status": status,
            "satisfied": sat,
            "failed": failed,
            "configuration": normalize_parameters(exp.parameters),
            "fingerprint": exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters}),
            "hypothesis": exp.hypothesis,
        })
    return rows


def best_valid_experiment(memory: ResearchMemory) -> dict[str, Any] | None:
    exp_id = memory.best_experiment_id
    if not exp_id:
        return None
    for exp in memory.experiments:
        if exp.id != exp_id:
            continue
        obs = _observation_from_experiment(exp)
        return {
            "experiment_id": exp.id,
            "parent_id": exp.parent_id,
            "hypothesis": exp.hypothesis,
            "configuration": normalize_parameters(exp.parameters),
            "observed_metrics": {k: v for k, v in obs.items() if isinstance(v, (int, float))},
            "status": "constraint-satisfying (may still miss the primary metric)",
        }
    return None


def build_failure_context(memory: ResearchMemory) -> PlannerFailureContext | None:
    failure = memory.failures[-1] if memory.failures else None
    exp = None
    if failure and failure.experiment_id:
        exp = next((e for e in memory.experiments if e.id == failure.experiment_id), None)
    if exp is None and memory.experiments:
        exp = memory.experiments[-1]

    obs = _observation_from_experiment(exp) if exp else {}
    sat, failed, violated = criteria_breakdown(memory.goal, obs) if exp else ([], [], [])

    # Constraint-violating latest run is the actionable planning context, even
    # when the newest Failure row is an older tool_crash.
    if memory.experiments:
        last = memory.experiments[-1]
        last_obs = _observation_from_experiment(last)
        last_sat, last_failed, last_violated = criteria_breakdown(memory.goal, last_obs)
        if last_violated:
            exp, obs, sat, failed, violated = last, last_obs, last_sat, last_failed, last_violated

    failure_type: FailureType | None = failure.failure_type if failure else None
    if violated:
        failure_type = FailureType.GOAL_DRIFT
    if failure_type is None:
        return None

    target: dict[str, float] = {}
    if memory.goal.target_value is not None:
        target[memory.goal.primary_metric] = memory.goal.target_value
    if any("latency" in c.name.lower() for c in memory.goal.constraints):
        target["latency_ms"] = LATENCY_LIMIT_MS
    previous = normalize_parameters(exp.parameters) if exp else {}
    required = None
    intervention = None
    for name in violated:
        spec = CONSTRAINT_INTERVENTIONS.get(name)
        if spec:
            required = spec.get("required_change")
            intervention = spec
            break
    if failure_type == FailureType.GOAL_DRIFT and not required and violated:
        required = "preserve satisfied criteria; change parameters that affect the violated constraint"
    elif failure_type == FailureType.REGRESSION:
        required = "do not retry the regressed configuration; branch from best_valid_experiment"
    elif failure_type == FailureType.RESULT_INCONSISTENCY:
        required = "do not rerun the same experiment; inspect independent evaluation evidence first"
    recovery = memory.recoveries[-1] if memory.recoveries else None
    evidence = list(recovery.evidence_used) if recovery else []
    if exp:
        evidence.append(f"{exp.id} fingerprint={configuration_fingerprint({'parameters': exp.parameters})}")
    diagnosis = (recovery.diagnosis if recovery else None)
    if failure:
        diagnosis = diagnosis or failure.description
    return PlannerFailureContext(
        failure_type=failure_type,
        experiment_id=(failure.experiment_id if failure else None) or (exp.id if exp else None),
        violated_constraints=violated,
        satisfied_criteria=sat,
        failed_criteria=failed,
        observed_metrics={k: float(v) for k, v in obs.items() if isinstance(v, (int, float))},
        target_metrics=target,
        previous_configuration=previous,
        rejected_configurations=rejected_configuration_rows(memory),
        diagnosis=diagnosis,
        evidence=evidence,
        required_change=required,
        intervention=intervention,
    )


def assess_proposal(memory: ResearchMemory, tool_name: str, tool_args: dict[str, Any]) -> ProposalAssessment:
    if tool_name != "run_experiment":
        return ProposalAssessment()
    fp = configuration_fingerprint(tool_args)
    if fp in rejected_fingerprints(memory):
        streak = memory.no_progress_count + 1
        return ProposalAssessment(
            blocked=True,
            terminal=streak >= MAX_NO_PROGRESS,
            reason=f"NO_PROGRESS: configuration {fp} was already rejected",
            fingerprint=fp,
        )
    ctx = build_failure_context(memory)
    latency_failed = bool(
        ctx
        and any("latency" in item.lower() for item in (*ctx.failed_criteria, *ctx.violated_constraints))
    )
    if ctx and ctx.failure_type == FailureType.GOAL_DRIFT and latency_failed:
        prev_score = complexity_score(ctx.previous_configuration)
        new_score = complexity_score(tool_args)
        if new_score >= prev_score:
            streak = memory.no_progress_count + 1
            return ProposalAssessment(
                blocked=True,
                terminal=streak >= MAX_NO_PROGRESS,
                reason=(
                    f"NO_PROGRESS: latency drift requires reduced complexity "
                    f"(score {new_score} >= previous {prev_score})"
                ),
                fingerprint=fp,
            )
    return ProposalAssessment(fingerprint=fp)


def record_blocked_proposal(memory: ResearchMemory, assessment: ProposalAssessment, tool_args: dict[str, Any]) -> None:
    memory.no_progress_count += 1
    memory.blocked_proposals.append({
        "fingerprint": assessment.fingerprint,
        "configuration": normalize_parameters(experiment_parameters(tool_args)),
        "reason": assessment.reason,
        "tool_args": tool_args,
    })
    if assessment.terminal:
        memory.unresolved_questions.append(assessment.reason)
