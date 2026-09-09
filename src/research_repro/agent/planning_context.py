"""Structured planner context: failures, fingerprints, and proposal guards.

The evaluator remains authoritative. This module turns evaluator facts into
actionable planner input and blocks silent repeats of tested or rejected configs.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from ..memory.models import (
    ConfigurationOutcome,
    ConfigurationRecord,
    Experiment,
    FailureType,
    GoalContract,
    ProgressKind,
    ResearchMemory,
)


# Defaults of environments/image_classification/train.py so {} fingerprints
# as the same configuration the sandbox actually executes.
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

# Experiment-affecting parameters only. IDs, aliases, and formatting do not count.
FINGERPRINT_KEYS = (
    "hidden_size",
    "hidden_layers",
    "max_iter",
    "lr",
    "batch_size",
    "normalize",
    "pca_components",
    "solver",
)

PARAM_ALIASES = {
    "learning_rate": "lr",
    "learning-rate": "lr",
    "eta": "lr",
    "hidden_layer_size": "hidden_size",
    "n_hidden": "hidden_size",
    "n_layers": "hidden_layers",
    "num_layers": "hidden_layers",
    "hidden_layer_count": "hidden_layers",
    "epochs": "max_iter",
    "n_iter": "max_iter",
    "n_iterations": "max_iter",
    "max_iterations": "max_iter",
    "batch": "batch_size",
    "pca": "pca_components",
    "n_components": "pca_components",
    "n_pca": "pca_components",
}

VALID_EXPERIMENT_PARAMS = {
    "hidden_size",
    "hidden_layers",
    "max_iter",
    "lr",
    "batch_size",
    "normalize",
    "pca_components",
    "solver",
    "seed",
    "latency_budget_ms",
}

VALID_SOLVERS = {"adam", "sgd", "lbfgs"}

PARAM_BOUNDS: dict[str, tuple[float, float]] = {
    "hidden_size": (1, 2048),
    "hidden_layers": (1, 8),
    "max_iter": (1, 2000),
    "lr": (1e-8, 1.0),
    "batch_size": (1, 10000),
    "pca_components": (0, 784),
    "seed": (0, 2**31 - 1),
    "latency_budget_ms": (1, 10_000),
}

INT_PARAMS = {
    "hidden_size",
    "hidden_layers",
    "max_iter",
    "batch_size",
    "pca_components",
    "seed",
}

# Domain knowledge: classes of intervention, not a prescribed configuration.
CONSTRAINT_INTERVENTIONS: dict[str, dict[str, Any]] = {
    "latency_under_100ms": {
        "reduce": ["hidden_size", "hidden_layers"],
        "consider": ["pca_components"],
        "preserve": ["normalize"],
        "required_change": "reduce computational cost / model complexity; preserve satisfied accuracy",
    },
}

CRITERION_INTERVENTIONS: dict[str, dict[str, Any]] = {
    "latency": {
        "interventions": [
            "reduce hidden_size",
            "reduce hidden_layers",
            "reduce model complexity",
            "reduce dimensionality",
        ],
        "required_change": "reduce computational cost / model complexity; preserve satisfied accuracy",
    },
    "accuracy": {
        "interventions": [
            "enable normalization",
            "increase training capacity",
            "increase max_iter",
            "adjust learning rate",
            "increase useful feature representation",
        ],
        "required_change": "improve predictive performance without violating latency <= 100ms",
    },
}

GENERIC_HYPOTHESIS_MARKERS = (
    "should be evaluated",
    "this configuration may work",
    "may work",
    "will be tested",
    "will be evaluated",
    "the proposed experiment",
    "a new configuration",
    "a distinct configuration",
    "advance the goal",
    "using a new experiment",
)

LATENCY_LIMIT_MS = 100.0
MAX_NO_PROGRESS = 3
METRIC_EPS = 1e-4


class PlannerFailureContext(BaseModel):
    failure_type: FailureType
    experiment_id: str | None = None
    violated_constraints: list[str] = Field(default_factory=list)
    satisfied_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    observed_metrics: dict[str, float] = Field(default_factory=dict)
    target_metrics: dict[str, float] = Field(default_factory=dict)
    previous_configuration: dict[str, Any] = Field(default_factory=dict)
    previous_experiment: dict[str, Any] | None = None
    previous_hypothesis: str = ""
    rejected_configurations: list[dict[str, Any]] = Field(default_factory=list)
    tested_configurations: list[dict[str, Any]] = Field(default_factory=list)
    best_valid_experiment: dict[str, Any] | None = None
    best_observed_accuracy: dict[str, Any] | None = None
    criterion_status: dict[str, dict[str, Any]] = Field(default_factory=dict)
    blocking_criterion: str | None = None
    diagnosis: str | None = None
    evidence: list[str] = Field(default_factory=list)
    required_change: str | None = None
    intervention: dict[str, Any] | None = None
    interventions: list[str] = Field(default_factory=list)
    guidance: str = ""


class PlanningEvidence(BaseModel):
    goal: str
    criteria: list[dict[str, Any]] = Field(default_factory=list)
    satisfied_criteria: list[str] = Field(default_factory=list)
    failed_criteria: list[str] = Field(default_factory=list)
    observed_metrics: dict[str, float] = Field(default_factory=dict)
    target_metrics: dict[str, float] = Field(default_factory=dict)
    constraint_status: str = "unknown"
    best_valid_experiment: dict[str, Any] | None = None
    best_observed_accuracy: dict[str, Any] | None = None
    previous_experiment: dict[str, Any] | None = None
    previous_configuration: dict[str, Any] = Field(default_factory=dict)
    previous_hypothesis: str = ""
    tested_configurations: list[dict[str, Any]] = Field(default_factory=list)
    rejected_configurations: list[dict[str, Any]] = Field(default_factory=list)
    recent_failures: list[dict[str, Any]] = Field(default_factory=list)
    required_change: str = ""
    blocking_criterion: str | None = None
    interventions: list[str] = Field(default_factory=list)
    guidance: str = ""
    criterion_status: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ProposalAssessment(BaseModel):
    blocked: bool = False
    terminal: bool = False
    reason: str = ""
    fingerprint: str = ""
    block_code: str | None = None
    previous_experiment: str | None = None
    previous_outcome: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    canonical_args: dict[str, Any] = Field(default_factory=dict)
    progress_kind: str | None = None


def experiment_parameters(args: dict[str, Any] | None) -> dict[str, Any]:
    if not args:
        return {}
    params = args.get("parameters") if isinstance(args.get("parameters"), dict) else args
    return dict(params) if isinstance(params, dict) else {}


def alias_key(key: str) -> str:
    raw = str(key).strip()
    lowered = raw.replace("-", "_").lower()
    if lowered in PARAM_ALIASES:
        return PARAM_ALIASES[lowered]
    dashed = raw.replace("_", "-").lower()
    if dashed in PARAM_ALIASES:
        return PARAM_ALIASES[dashed]
    if lowered in VALID_EXPERIMENT_PARAMS:
        return lowered
    return raw


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_number(key: str, value: Any) -> Any:
    if value is None:
        return None
    if key in INT_PARAMS:
        return int(float(value))
    if key in {"lr", "latency_budget_ms"}:
        return float(value)
    return value


def canonicalize_parameters(params: dict[str, Any] | None) -> dict[str, Any]:
    """Map aliases and types onto the environment's real parameter names."""
    out: dict[str, Any] = {}
    if not params:
        return out
    for raw_key, value in params.items():
        if value is None:
            continue
        key = alias_key(str(raw_key))
        if key == "normalize":
            out[key] = _as_bool(value)
        elif key == "solver":
            out[key] = str(value).strip().lower()
        elif key in PARAM_BOUNDS or key in INT_PARAMS:
            try:
                out[key] = _as_number(key, value)
            except (TypeError, ValueError):
                out[key] = value
        else:
            out[key] = value
    return out


def canonicalize_tool_args(args: dict[str, Any] | None) -> dict[str, Any]:
    """Return a copy of tool args with canonical experiment parameters."""
    raw = dict(args or {})
    params = experiment_parameters(raw)
    # Promote accidental top-level aliases into parameters.
    for key in list(raw.keys()):
        mapped = alias_key(key)
        if mapped in VALID_EXPERIMENT_PARAMS and key not in {"parameters", "experiment_id", "timeout_seconds", "seed", "parent_experiment_id"}:
            if mapped not in params:
                params[mapped] = raw.pop(key)
            else:
                raw.pop(key, None)
    canonical_params = canonicalize_parameters(params)
    raw["parameters"] = canonical_params
    if "seed" in canonical_params and "seed" not in raw:
        raw["seed"] = canonical_params["seed"]
    return raw


def normalize_parameters(params: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_TRAIN_PARAMS)
    canonical = canonicalize_parameters(experiment_parameters(params) if params and "parameters" in (params or {}) else params)
    if not canonical and params:
        canonical = canonicalize_parameters(params)
    merged.update({k: v for k, v in canonical.items() if v is not None})
    merged["normalize"] = _as_bool(merged.get("normalize", False))
    if "solver" in merged:
        merged["solver"] = str(merged["solver"]).strip().lower()
    return merged


def configuration_fingerprint(args: dict[str, Any] | None) -> str:
    norm = normalize_parameters(experiment_parameters(args) or args)
    parts: list[str] = []
    for key in FINGERPRINT_KEYS:
        value = norm.get(key)
        if key == "normalize":
            value = bool(value)
        elif key in INT_PARAMS and value is not None:
            try:
                value = int(float(value))
            except (TypeError, ValueError):
                pass
        elif key == "lr" and value is not None:
            try:
                value = float(value)
                value = float(f"{value:.8g}")
            except (TypeError, ValueError):
                pass
        parts.append(f"{key}={value}")
    return "|".join(parts)


def material_param_delta(previous: dict[str, Any] | None, proposed: dict[str, Any] | None) -> dict[str, tuple[Any, Any]]:
    prev = normalize_parameters(previous)
    new = normalize_parameters(proposed)
    changed: dict[str, tuple[Any, Any]] = {}
    for key in FINGERPRINT_KEYS:
        if prev.get(key) != new.get(key):
            changed[key] = (prev.get(key), new.get(key))
    return changed


def is_materially_different(previous: dict[str, Any] | None, proposed: dict[str, Any] | None) -> bool:
    return bool(material_param_delta(previous, proposed))


def complexity_score(params: dict[str, Any] | None) -> int:
    norm = normalize_parameters(experiment_parameters(params) or params)
    try:
        return int(norm.get("hidden_size", 128)) * int(norm.get("hidden_layers", 2))
    except (TypeError, ValueError):
        return 256


def validate_configuration(params: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    canonical = canonicalize_parameters(params or {})
    for key in canonical:
        if key not in VALID_EXPERIMENT_PARAMS:
            errors.append(f"unknown experiment parameter: {key}")
    solver = canonical.get("solver")
    if solver is not None and str(solver).lower() not in VALID_SOLVERS:
        errors.append(f"unsupported solver '{solver}'; expected one of {sorted(VALID_SOLVERS)}")
    for key, (lo, hi) in PARAM_BOUNDS.items():
        if key not in canonical:
            continue
        try:
            value = float(canonical[key])
        except (TypeError, ValueError):
            errors.append(f"malformed type for {key}: {canonical[key]!r}")
            continue
        if value < lo or value > hi:
            errors.append(f"impossible value for {key}: {canonical[key]} (allowed {lo}..{hi})")
    if "normalize" in canonical and not isinstance(canonical["normalize"], bool):
        errors.append(f"malformed type for normalize: {canonical['normalize']!r}")
    return errors


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


def target_metrics(goal: GoalContract) -> dict[str, float]:
    target: dict[str, float] = {}
    if goal.criterion and goal.criterion.minimum is not None:
        target[goal.primary_metric] = goal.criterion.minimum
    elif goal.criterion and goal.criterion.target is not None:
        target[goal.primary_metric] = goal.criterion.target
    elif goal.target_value is not None:
        target[goal.primary_metric] = goal.target_value
    if any("latency" in c.name.lower() for c in goal.constraints):
        target["latency_ms"] = LATENCY_LIMIT_MS
    return target


def criterion_status_map(goal: GoalContract, observation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    targets = target_metrics(goal)
    status: dict[str, dict[str, Any]] = {}
    metric = goal.primary_metric
    observed = observation.get(metric)
    if observed is None and metric == "accuracy":
        observed = observation.get("accuracy")
    metric_target = targets.get(metric)
    if isinstance(observed, (int, float)):
        passed = goal.is_achieved(float(observed))
        status[metric] = {
            "observed": float(observed),
            "target": metric_target,
            "status": "PASS" if passed else "FAIL",
        }
    elif metric_target is not None:
        status[metric] = {"observed": None, "target": metric_target, "status": "UNKNOWN"}
    if "latency_ms" in targets:
        latency = observation.get("latency_ms")
        if latency is None:
            nested = observation.get("additional_metrics") if isinstance(observation.get("additional_metrics"), dict) else {}
            latency = nested.get("latency_ms") if nested else None
        if isinstance(latency, (int, float)):
            status["latency"] = {
                "observed": float(latency),
                "target": LATENCY_LIMIT_MS,
                "status": "PASS" if float(latency) <= LATENCY_LIMIT_MS else "FAIL",
            }
        else:
            status["latency"] = {"observed": None, "target": LATENCY_LIMIT_MS, "status": "UNKNOWN"}
    return status


def criteria_breakdown(goal: GoalContract, observation: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    satisfied: list[str] = []
    failed: list[str] = []
    violated: list[str] = []
    status = criterion_status_map(goal, observation)
    metric_label = _metric_criterion_label(goal)
    metric_row = status.get(goal.primary_metric) or status.get("accuracy")
    if metric_row:
        if metric_row["status"] == "PASS":
            satisfied.append(metric_label)
        elif metric_row["status"] == "FAIL":
            failed.append(metric_label)
    latency_row = status.get("latency")
    if latency_row:
        label = f"latency <= {int(LATENCY_LIMIT_MS)}ms"
        if latency_row["status"] == "PASS":
            satisfied.append(label)
        elif latency_row["status"] == "FAIL":
            failed.append(label)
            for constraint in goal.constraints:
                if "latency" in constraint.name.lower() and constraint.name not in violated:
                    violated.append(constraint.name)
    for constraint in goal.constraints:
        name = constraint.name
        if "latency" in name.lower():
            continue
        if not goal.constraints_hold(observation) and name not in violated:
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


def blocking_criterion(goal: GoalContract, observation: dict[str, Any]) -> str | None:
    status = criterion_status_map(goal, observation)
    latency = status.get("latency")
    accuracy = status.get(goal.primary_metric) or status.get("accuracy")
    latency_fail = bool(latency and latency.get("status") == "FAIL")
    accuracy_fail = bool(accuracy and accuracy.get("status") == "FAIL")
    if latency_fail and not accuracy_fail:
        return "latency"
    if accuracy_fail and not latency_fail:
        return "accuracy"
    if latency_fail and accuracy_fail:
        return "both"
    return None


def required_change_for(goal: GoalContract, observation: dict[str, Any]) -> tuple[str, list[str]]:
    blocking = blocking_criterion(goal, observation)
    if blocking == "latency":
        spec = CRITERION_INTERVENTIONS["latency"]
        return spec["required_change"], list(spec["interventions"])
    if blocking == "accuracy":
        spec = CRITERION_INTERVENTIONS["accuracy"]
        return spec["required_change"], list(spec["interventions"])
    if blocking == "both":
        return (
            "consider accuracy/latency tradeoffs; do not blindly optimize one metric while ignoring the other",
            list(CRITERION_INTERVENTIONS["latency"]["interventions"])
            + list(CRITERION_INTERVENTIONS["accuracy"]["interventions"]),
        )
    return "propose a materially different configuration grounded in observed evidence", []


def guidance_for(goal: GoalContract, observation: dict[str, Any]) -> str:
    blocking = blocking_criterion(goal, observation)
    status = criterion_status_map(goal, observation)
    acc = status.get(goal.primary_metric) or status.get("accuracy") or {}
    lat = status.get("latency") or {}
    if blocking == "accuracy":
        return (
            "Latency is currently satisfied. "
            "Accuracy is currently the blocking criterion. "
            f"The next proposal must primarily target accuracy improvement while preserving latency <= {int(LATENCY_LIMIT_MS)}ms."
        )
    if blocking == "latency":
        return (
            "Accuracy is currently satisfied. "
            "Latency is currently the blocking criterion. "
            "The next proposal must reduce computational cost while preserving as much accuracy as possible."
        )
    if blocking == "both":
        return (
            "Both accuracy and latency currently fail. "
            "The next proposal must consider the tradeoff and must not optimize one metric while ignoring the other."
        )
    if acc.get("status") == "PASS" and lat.get("status") == "PASS":
        return "All recorded criteria currently pass. Do not declare success; wait for evaluator verification."
    return "Use observed metrics to choose a materially different next configuration."


def experiment_is_rejected(memory: ResearchMemory, exp: Experiment) -> bool:
    obs = _observation_from_experiment(exp)
    if exp.evaluation_status.value == "negative" and not memory.goal.constraints_hold(obs):
        return True
    for failure in memory.failures:
        if failure.experiment_id == exp.id and failure.failure_type in (
            FailureType.GOAL_DRIFT,
            FailureType.REGRESSION,
            FailureType.RESULT_INCONSISTENCY,
        ):
            return True
    fp = exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters})
    if fp in memory.rejected_configs:
        return True
    return False


def rejected_fingerprints(memory: ResearchMemory) -> set[str]:
    out: set[str] = set(memory.rejected_configs.keys())
    for exp in memory.experiments:
        if experiment_is_rejected(memory, exp):
            out.add(exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters}))
    for blocked in memory.blocked_proposals:
        fp = blocked.get("fingerprint")
        if fp:
            out.add(str(fp))
    for fp, rec in memory.tested_configs.items():
        if rec.outcome in {
            ConfigurationOutcome.CONSTRAINT_VIOLATION,
            ConfigurationOutcome.INVALID_CONFIGURATION,
        }:
            out.add(fp)
    return out


def tested_fingerprints(memory: ResearchMemory) -> set[str]:
    out = set(memory.tested_configs.keys())
    for exp in memory.experiments:
        if exp.observed_result is not None:
            out.add(exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters}))
    return out


def rejected_configuration_rows(memory: ResearchMemory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for exp in memory.experiments:
        if not experiment_is_rejected(memory, exp):
            continue
        obs = _observation_from_experiment(exp)
        _sat, failed, violated = criteria_breakdown(memory.goal, obs)
        fp = exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters})
        seen.add(fp)
        rows.append({
            "experiment_id": exp.id,
            "fingerprint": fp,
            "configuration": normalize_parameters(exp.parameters),
            "observed_metrics": {k: v for k, v in obs.items() if isinstance(v, (int, float))},
            "reason": memory.rejected_configs.get(fp) or ", ".join(violated or failed) or "rejected",
        })
    for fp, reason in memory.rejected_configs.items():
        if fp in seen:
            continue
        rec = memory.tested_configs.get(fp)
        rows.append({
            "fingerprint": fp,
            "configuration": rec.config if rec else {},
            "reason": reason,
            "experiment_id": rec.experiment_id if rec else None,
        })
    for blocked in memory.blocked_proposals:
        rows.append(blocked)
    return rows


def tested_configuration_rows(memory: ResearchMemory) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fp, rec in memory.tested_configs.items():
        rows.append({
            "fingerprint": fp,
            "configuration": rec.config,
            "experiment_id": rec.experiment_id,
            "accuracy": rec.accuracy,
            "latency_ms": rec.latency_ms,
            "outcome": rec.outcome.value,
            "constraint_status": rec.constraint_status,
            "criterion_status": rec.criterion_status,
            "failure_type": rec.failure_type.value if rec.failure_type else None,
            "hypothesis": rec.hypothesis,
        })
    if rows:
        return rows
    for exp in memory.experiments:
        if not exp.observed_result:
            continue
        obs = _observation_from_experiment(exp)
        rows.append({
            "fingerprint": exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters}),
            "configuration": normalize_parameters(exp.parameters),
            "experiment_id": exp.id,
            "accuracy": obs.get("accuracy"),
            "latency_ms": obs.get("latency_ms"),
            "hypothesis": exp.hypothesis,
        })
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
            "parent_id": exp.parent_id,
            "reason": exp.reason,
            "trigger": exp.trigger,
        })
    return rows


def _experiment_row(memory: ResearchMemory, exp: Experiment) -> dict[str, Any]:
    obs = _observation_from_experiment(exp)
    sat, failed, violated = criteria_breakdown(memory.goal, obs)
    return {
        "experiment_id": exp.id,
        "parent_id": exp.parent_id,
        "hypothesis": exp.hypothesis,
        "reason": exp.reason,
        "trigger": exp.trigger,
        "configuration": normalize_parameters(exp.parameters),
        "observed_metrics": {k: v for k, v in obs.items() if isinstance(v, (int, float))},
        "satisfied_criteria": sat,
        "failed_criteria": failed,
        "violated_constraints": violated,
        "fingerprint": exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters}),
    }


def best_valid_experiment(memory: ResearchMemory) -> dict[str, Any] | None:
    exp_id = memory.best_experiment_id
    if not exp_id:
        return None
    for exp in memory.experiments:
        if exp.id != exp_id:
            continue
        row = _experiment_row(memory, exp)
        row["status"] = "constraint-satisfying (may still miss the primary metric)"
        return row
    return None


def best_observed_accuracy(memory: ResearchMemory) -> dict[str, Any] | None:
    exp_id = memory.best_observed_accuracy_id
    if not exp_id:
        # Fall back to highest accuracy including illegal runs.
        best = None
        best_acc = None
        for exp in memory.experiments:
            if not exp.observed_result or exp.observed_result.accuracy is None:
                continue
            if best_acc is None or exp.observed_result.accuracy > best_acc:
                best_acc = exp.observed_result.accuracy
                best = exp
        if best is None:
            return None
        row = _experiment_row(memory, best)
        row["status"] = "highest observed accuracy (may violate constraints)"
        return row
    for exp in memory.experiments:
        if exp.id != exp_id:
            continue
        row = _experiment_row(memory, exp)
        row["status"] = "highest observed accuracy (may violate constraints)"
        return row
    return None


def previous_experiment(memory: ResearchMemory) -> Experiment | None:
    if not memory.experiments:
        return None
    return memory.experiments[-1]


def _retry_allowed(memory: ResearchMemory, fingerprint: str) -> bool:
    """Same configuration may be retried only for recorded non-determinism."""
    if not memory.failures:
        return False
    last = memory.failures[-1]
    if last.failure_type != FailureType.NON_DETERMINISM:
        return False
    rec = memory.tested_configs.get(fingerprint)
    return rec is not None and rec.outcome != ConfigurationOutcome.INVALID_CONFIGURATION


def build_planning_evidence(memory: ResearchMemory) -> PlanningEvidence:
    exp = previous_experiment(memory)
    obs = _observation_from_experiment(exp) if exp else {}
    sat, failed, violated = criteria_breakdown(memory.goal, obs) if exp else ([], [], [])
    required, interventions = required_change_for(memory.goal, obs) if exp else ("", [])
    guidance = guidance_for(memory.goal, obs) if exp else "Start from the default configuration and gather evidence."
    blocking = blocking_criterion(memory.goal, obs) if exp else None
    constraint_status = "violated" if violated else ("all_met" if exp else "unknown")
    prev_row = _experiment_row(memory, exp) if exp else None
    return PlanningEvidence(
        goal=memory.goal.objective,
        criteria=[
            {"name": name, **row}
            for name, row in criterion_status_map(memory.goal, obs).items()
        ],
        satisfied_criteria=sat,
        failed_criteria=failed,
        observed_metrics={k: float(v) for k, v in obs.items() if isinstance(v, (int, float))},
        target_metrics=target_metrics(memory.goal),
        constraint_status=constraint_status,
        best_valid_experiment=best_valid_experiment(memory),
        best_observed_accuracy=best_observed_accuracy(memory),
        previous_experiment=prev_row,
        previous_configuration=normalize_parameters(exp.parameters) if exp else dict(DEFAULT_TRAIN_PARAMS),
        previous_hypothesis=exp.hypothesis if exp else "",
        tested_configurations=tested_configuration_rows(memory),
        rejected_configurations=rejected_configuration_rows(memory),
        recent_failures=[
            {
                "failure_type": f.failure_type.value,
                "description": f.description,
                "experiment_id": f.experiment_id,
                "recovered": f.recovered,
            }
            for f in memory.recent_failures(5)
        ],
        required_change=required,
        blocking_criterion=blocking,
        interventions=interventions,
        guidance=guidance,
        criterion_status=criterion_status_map(memory.goal, obs),
    )


def build_failure_context(memory: ResearchMemory) -> PlannerFailureContext | None:
    failure = memory.failures[-1] if memory.failures else None
    exp = None
    if failure and failure.experiment_id:
        exp = next((e for e in memory.experiments if e.id == failure.experiment_id), None)
    if exp is None and memory.experiments:
        exp = memory.experiments[-1]

    obs = _observation_from_experiment(exp) if exp else {}
    sat, failed, violated = criteria_breakdown(memory.goal, obs) if exp else ([], [], [])

    if memory.experiments:
        last = memory.experiments[-1]
        last_obs = _observation_from_experiment(last)
        last_sat, last_failed, last_violated = criteria_breakdown(memory.goal, last_obs)
        if last_violated:
            exp, obs, sat, failed, violated = last, last_obs, last_sat, last_failed, last_violated

    failure_type: FailureType | None = failure.failure_type if failure else None
    if violated:
        failure_type = FailureType.GOAL_DRIFT
    evidence_pack = build_planning_evidence(memory)
    if failure_type is None and not evidence_pack.failed_criteria:
        return None
    if failure_type is None:
        # Objective-only miss is still planner evidence; no taxonomy failure required.
        failure_type = FailureType.GOAL_DRIFT if violated else FailureType.REGRESSION

    previous = normalize_parameters(exp.parameters) if exp else {}
    required, interventions = required_change_for(memory.goal, obs)
    intervention = None
    for name in violated:
        spec = CONSTRAINT_INTERVENTIONS.get(name)
        if spec:
            intervention = spec
            break
    if failure_type == FailureType.REGRESSION and not violated:
        required = "do not retry the regressed configuration; branch from best_valid_experiment"
    elif failure_type == FailureType.RESULT_INCONSISTENCY:
        required = "do not rerun the same experiment; inspect independent evaluation evidence first"
    recovery = memory.recoveries[-1] if memory.recoveries else None
    evidence = list(recovery.evidence_used) if recovery else []
    if exp:
        evidence.append(f"{exp.id} fingerprint={configuration_fingerprint({'parameters': exp.parameters})}")
    diagnosis = recovery.diagnosis if recovery else None
    if failure:
        diagnosis = diagnosis or failure.description
    return PlannerFailureContext(
        failure_type=failure_type,
        experiment_id=(failure.experiment_id if failure else None) or (exp.id if exp else None),
        violated_constraints=violated,
        satisfied_criteria=sat,
        failed_criteria=failed,
        observed_metrics={k: float(v) for k, v in obs.items() if isinstance(v, (int, float))},
        target_metrics=target_metrics(memory.goal),
        previous_configuration=previous,
        previous_experiment=_experiment_row(memory, exp) if exp else None,
        previous_hypothesis=exp.hypothesis if exp else "",
        rejected_configurations=rejected_configuration_rows(memory),
        tested_configurations=tested_configuration_rows(memory),
        best_valid_experiment=best_valid_experiment(memory),
        best_observed_accuracy=best_observed_accuracy(memory),
        criterion_status=criterion_status_map(memory.goal, obs),
        blocking_criterion=blocking_criterion(memory.goal, obs),
        diagnosis=diagnosis,
        evidence=evidence,
        required_change=required,
        intervention=intervention,
        interventions=interventions,
        guidance=guidance_for(memory.goal, obs),
    )


def assess_proposal(memory: ResearchMemory, tool_name: str, tool_args: dict[str, Any]) -> ProposalAssessment:
    if tool_name != "run_experiment":
        return ProposalAssessment()
    canonical = canonicalize_tool_args(tool_args)
    fp = configuration_fingerprint(canonical)
    errors = validate_configuration(canonical.get("parameters") or {})
    if errors:
        streak = memory.no_progress_count + 1
        return ProposalAssessment(
            blocked=True,
            terminal=streak >= MAX_NO_PROGRESS,
            reason="INVALID_CONFIGURATION: " + "; ".join(errors),
            fingerprint=fp,
            block_code="invalid_configuration",
            validation_errors=errors,
            canonical_args=canonical,
            progress_kind=ProgressKind.INVALID_ACTION_REPEATED.value,
        )

    tested_rec = memory.tested_configs.get(fp)
    if tested_rec is None:
        for exp in memory.experiments:
            exp_fp = exp.config_fingerprint or configuration_fingerprint({"parameters": exp.parameters})
            if exp_fp == fp and exp.observed_result is not None:
                tested_rec = ConfigurationRecord(
                    fingerprint=fp,
                    config=normalize_parameters(exp.parameters),
                    experiment_id=exp.id,
                    accuracy=exp.observed_result.accuracy,
                    latency_ms=exp.observed_result.additional_metrics.get("latency_ms"),
                    outcome=(
                        ConfigurationOutcome.CONSTRAINT_VIOLATION
                        if experiment_is_rejected(memory, exp)
                        else ConfigurationOutcome.OBJECTIVE_FAILURE
                    ),
                    hypothesis=exp.hypothesis,
                )
                break

    if tested_rec is not None and not _retry_allowed(memory, fp):
        streak = memory.no_progress_count + 1
        return ProposalAssessment(
            blocked=True,
            terminal=streak >= MAX_NO_PROGRESS,
            reason=(
                f"NO_PROGRESS: proposal_blocked reason=duplicate_tested_configuration "
                f"fingerprint={fp} previous_experiment={tested_rec.experiment_id} "
                f"previous_outcome={tested_rec.outcome.value}"
            ),
            fingerprint=fp,
            block_code="duplicate_tested_configuration",
            previous_experiment=tested_rec.experiment_id,
            previous_outcome=tested_rec.outcome.value,
            canonical_args=canonical,
            progress_kind=ProgressKind.DUPLICATE_CONFIGURATION.value,
        )

    if fp in rejected_fingerprints(memory) and not _retry_allowed(memory, fp):
        streak = memory.no_progress_count + 1
        return ProposalAssessment(
            blocked=True,
            terminal=streak >= MAX_NO_PROGRESS,
            reason=f"NO_PROGRESS: configuration {fp} was already rejected",
            fingerprint=fp,
            block_code="duplicate_rejected_configuration",
            canonical_args=canonical,
            progress_kind=ProgressKind.IDENTICAL_REJECTED_PROPOSAL.value,
        )

    ctx = build_failure_context(memory)
    latency_failed = bool(
        ctx
        and any("latency" in item.lower() for item in (*ctx.failed_criteria, *ctx.violated_constraints))
    )
    accuracy_ok = bool(ctx and any("accuracy" in item.lower() for item in ctx.satisfied_criteria))
    if ctx and ctx.failure_type == FailureType.GOAL_DRIFT and latency_failed and accuracy_ok:
        prev_score = complexity_score(ctx.previous_configuration)
        new_score = complexity_score(canonical)
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
                block_code="no_meaningful_change",
                canonical_args=canonical,
                progress_kind=ProgressKind.SAME_FAILURE_WITHOUT_CHANGE.value,
            )
    return ProposalAssessment(fingerprint=fp, canonical_args=canonical)


def record_blocked_proposal(memory: ResearchMemory, assessment: ProposalAssessment, tool_args: dict[str, Any]) -> None:
    memory.no_progress_count += 1
    memory.last_progress_kind = assessment.progress_kind or ProgressKind.DUPLICATE_CONFIGURATION.value
    payload = {
        "fingerprint": assessment.fingerprint,
        "configuration": normalize_parameters(experiment_parameters(assessment.canonical_args or tool_args)),
        "reason": assessment.reason,
        "block_code": assessment.block_code,
        "previous_experiment": assessment.previous_experiment,
        "previous_outcome": assessment.previous_outcome,
        "tool_args": tool_args,
    }
    memory.blocked_proposals.append(payload)
    if assessment.fingerprint:
        memory.rejected_configs.setdefault(assessment.fingerprint, assessment.reason)
    if assessment.terminal:
        memory.terminate_reason = "no_progress"
        memory.unresolved_questions.append(assessment.reason)


def record_configuration_outcome(
    memory: ResearchMemory,
    args: dict[str, Any],
    *,
    experiment_id: str | None,
    accuracy: float | None,
    latency_ms: float | None,
    outcome: ConfigurationOutcome,
    hypothesis: str = "",
    failure_type: FailureType | None = None,
    constraint_status: str = "",
    criterion_status: dict[str, str] | None = None,
) -> ConfigurationRecord:
    canonical = canonicalize_tool_args(args)
    fp = configuration_fingerprint(canonical)
    record = ConfigurationRecord(
        fingerprint=fp,
        config=normalize_parameters(canonical.get("parameters") or {}),
        experiment_id=experiment_id,
        accuracy=accuracy,
        latency_ms=latency_ms,
        constraint_status=constraint_status,
        criterion_status=criterion_status or {},
        failure_type=failure_type,
        hypothesis=hypothesis,
        outcome=outcome,
    )
    memory.tested_configs[fp] = record
    if outcome in {
        ConfigurationOutcome.CONSTRAINT_VIOLATION,
        ConfigurationOutcome.INVALID_CONFIGURATION,
    }:
        memory.rejected_configs[fp] = outcome.value
    return record


def classify_progress(memory: ResearchMemory, args: dict[str, Any], observation: dict[str, Any] | None) -> ProgressKind:
    fp = configuration_fingerprint(canonicalize_tool_args(args))
    if fp in memory.tested_configs and memory.tested_configs[fp].experiment_id:
        return ProgressKind.DUPLICATE_CONFIGURATION
    obs = observation or {}
    prev = previous_experiment(memory)
    if prev and prev.observed_result:
        prev_obs = _observation_from_experiment(prev)
        changed = material_param_delta(prev.parameters, experiment_parameters(args) or args)
        if not changed:
            return ProgressKind.SAME_FAILURE_WITHOUT_CHANGE
        prev_failed = criteria_breakdown(memory.goal, prev_obs)[1]
        now_failed = criteria_breakdown(memory.goal, obs)[1]
        if any("latency" in item.lower() for item in prev_failed) and not any("latency" in item.lower() for item in now_failed):
            return ProgressKind.FIXED_FAILED_CONSTRAINT
        metric = memory.goal.primary_metric
        prev_m = prev_obs.get(metric)
        now_m = obs.get(metric)
        if isinstance(prev_m, (int, float)) and isinstance(now_m, (int, float)) and now_m > prev_m + METRIC_EPS:
            return ProgressKind.IMPROVED_OBJECTIVE
        if isinstance(prev_m, (int, float)) and isinstance(now_m, (int, float)) and abs(now_m - prev_m) <= METRIC_EPS:
            lat_prev = prev_obs.get("latency_ms")
            lat_now = obs.get("latency_ms")
            if (
                isinstance(lat_prev, (int, float))
                and isinstance(lat_now, (int, float))
                and abs(lat_now - lat_prev) <= 0.5
            ):
                return ProgressKind.NO_MEANINGFUL_METRIC_CHANGE
        if changed:
            return ProgressKind.MEANINGFUL_STRATEGY_CHANGE
    return ProgressKind.NEW_VALID_CONFIGURATION


def apply_progress(memory: ResearchMemory, kind: ProgressKind) -> None:
    memory.last_progress_kind = kind.value
    if kind in {
        ProgressKind.NEW_VALID_CONFIGURATION,
        ProgressKind.IMPROVED_OBJECTIVE,
        ProgressKind.FIXED_FAILED_CONSTRAINT,
        ProgressKind.NEW_EVIDENCE,
        ProgressKind.MEANINGFUL_STRATEGY_CHANGE,
    }:
        memory.no_progress_count = 0
    else:
        memory.no_progress_count += 1
        if memory.no_progress_count >= MAX_NO_PROGRESS:
            memory.terminate_reason = "no_progress"


def is_generic_hypothesis(text: str | None) -> bool:
    blob = (text or "").strip().lower()
    if not blob:
        return True
    if len(blob) < 48:
        return True
    return any(marker in blob for marker in GENERIC_HYPOTHESIS_MARKERS)


def evidence_based_hypothesis(memory: ResearchMemory, params: dict[str, Any] | None) -> str:
    evidence = build_planning_evidence(memory)
    delta = material_param_delta(evidence.previous_configuration, params)
    change = ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in delta.items()) or "a materially different configuration"
    acc = evidence.criterion_status.get(memory.goal.primary_metric) or evidence.criterion_status.get("accuracy") or {}
    lat = evidence.criterion_status.get("latency") or {}
    target_acc = acc.get("target")
    observed_acc = acc.get("observed")
    observed_lat = lat.get("observed")
    if evidence.blocking_criterion == "accuracy":
        return (
            f"Given the previous run observed accuracy={observed_acc} (target {target_acc}) "
            f"with legal latency={observed_lat}ms, {change} may recover accuracy while remaining "
            f"under the latency cap of {int(LATENCY_LIMIT_MS)}ms. Falsified if accuracy does not "
            f"improve or latency exceeds {int(LATENCY_LIMIT_MS)}ms."
        )
    if evidence.blocking_criterion == "latency":
        return (
            f"Given accuracy={observed_acc} already meets the target but latency={observed_lat}ms "
            f"exceeds {int(LATENCY_LIMIT_MS)}ms, {change} should reduce computational cost enough "
            f"to satisfy the latency cap while preserving as much accuracy as possible. "
            f"Falsified if latency remains above {int(LATENCY_LIMIT_MS)}ms."
        )
    if evidence.blocking_criterion == "both":
        return (
            f"Both accuracy={observed_acc} and latency={observed_lat}ms currently miss the contract. "
            f"{change} is a tradeoff intended to move both metrics toward "
            f"accuracy>={target_acc} and latency<={int(LATENCY_LIMIT_MS)}ms. "
            f"Falsified if either metric moves the wrong direction without a compensating gain."
        )
    return (
        f"The default/current evidence is accuracy={observed_acc}, latency={observed_lat}ms. "
        f"Testing {change} should produce a measurable change in the blocking metric. "
        f"Falsified if metrics are unchanged."
    )


def ensure_falsifiable_hypothesis(memory: ResearchMemory, hypothesis: str, params: dict[str, Any] | None) -> str:
    if is_generic_hypothesis(hypothesis):
        return evidence_based_hypothesis(memory, params)
    return hypothesis


_UNRECOGNIZED_RE = re.compile(r"unrecognized arguments:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def parse_unrecognized_cli_args(error: str) -> dict[str, Any]:
    match = _UNRECOGNIZED_RE.search(error or "")
    if not match:
        return {}
    tokens = match.group(1).strip().split()
    parsed: dict[str, Any] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if not token.startswith("--"):
            i += 1
            continue
        key = token[2:]
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
            parsed[key] = tokens[i + 1]
            i += 2
        else:
            parsed[key] = True
            i += 1
    return parsed


def repair_from_argparse_error(args: dict[str, Any], error: str) -> dict[str, Any]:
    """Map invalid CLI flags such as --learning-rate onto valid tool parameters."""
    repaired = canonicalize_tool_args(args)
    params = dict(repaired.get("parameters") or {})
    unrecognized = parse_unrecognized_cli_args(error)
    for raw_key, value in unrecognized.items():
        mapped = alias_key(raw_key)
        if mapped in VALID_EXPERIMENT_PARAMS:
            params[mapped] = value
            params.pop(raw_key, None)
            params.pop(raw_key.replace("-", "_"), None)
    # Also drop any still-invalid keys that aliases can replace.
    for key in list(params.keys()):
        mapped = alias_key(key)
        if mapped != key and mapped in VALID_EXPERIMENT_PARAMS:
            if mapped not in params:
                params[mapped] = params.pop(key)
            else:
                params.pop(key)
        elif key not in VALID_EXPERIMENT_PARAMS:
            # Leave unknown keys so validation can still reject them.
            pass
    repaired["parameters"] = canonicalize_parameters(params)
    return repaired


def action_was_repaired(original: dict[str, Any], repaired: dict[str, Any]) -> bool:
    orig_params = experiment_parameters(original)
    new_params = experiment_parameters(repaired)
    if canonicalize_parameters(orig_params) != canonicalize_parameters(new_params):
        return True
    orig_keys = {alias_key(k) for k in orig_params}
    new_keys = {alias_key(k) for k in new_params}
    if orig_keys != new_keys:
        return True
    orig_raw = json.dumps(orig_params, sort_keys=True, default=str)
    new_raw = json.dumps(new_params, sort_keys=True, default=str)
    return orig_raw != new_raw


def cli_flag_for(param: str) -> str:
    return f"--{param.replace('_', '-')}"


def deterministic_next_parameters(memory: ResearchMemory) -> dict[str, Any] | None:
    """Choose a legal next configuration from intervention classes and evidence.

    Does not hardcode a known-successful config. It only applies one class of
    change that addresses the blocking criterion, skipping fingerprints already
    tested or rejected.
    """
    evidence = build_planning_evidence(memory)
    base = dict(evidence.previous_configuration or DEFAULT_TRAIN_PARAMS)
    candidates: list[dict[str, Any]] = []
    blocking = evidence.blocking_criterion
    if blocking in {"latency", "both"}:
        layers = int(base.get("hidden_layers") or 2)
        size = int(base.get("hidden_size") or 128)
        if layers > 1:
            cand = dict(base)
            cand["hidden_layers"] = layers - 1
            candidates.append(cand)
        if size > 32:
            cand = dict(base)
            cand["hidden_size"] = max(32, size // 2)
            candidates.append(cand)
        if layers > 1 and size > 32:
            cand = dict(base)
            cand["hidden_layers"] = 1
            cand["hidden_size"] = max(32, size // 2)
            candidates.append(cand)
        if int(base.get("pca_components") or 0) == 0:
            cand = dict(base)
            cand["pca_components"] = 64
            candidates.append(cand)
    if blocking in {"accuracy", "both", None}:
        if not _as_bool(base.get("normalize", False)):
            cand = dict(base)
            cand["normalize"] = True
            candidates.append(cand)
        cand = dict(base)
        cand["max_iter"] = min(200, int(base.get("max_iter") or 100) + 50)
        candidates.append(cand)
        lr = float(base.get("lr") or 0.001)
        cand = dict(base)
        cand["lr"] = 0.0005 if lr >= 0.001 else 0.001
        candidates.append(cand)
        if blocking == "both" or (blocking == "accuracy" and not candidates):
            cand = dict(base)
            cand["normalize"] = True
            cand["max_iter"] = min(200, int(base.get("max_iter") or 100) + 50)
            candidates.append(cand)
    for cand in candidates:
        args = {"experiment_id": "candidate", "parameters": cand}
        if not assess_proposal(memory, "run_experiment", args).blocked:
            return canonicalize_parameters(cand)
    return None
