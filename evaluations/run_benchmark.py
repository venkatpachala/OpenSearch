"""Run the real self-correcting vs naive benchmark.

Both policies use the same goal, tools, environment, seed, budget, and
fault schedule. The only policy difference is post-evaluation behavior.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from research_repro.agent.loop import AgentLoop
from research_repro.agent.strategy import NaiveStrategy
from research_repro.config import Config
from research_repro.memory.models import Constraint, GoalContract, MetricCriterion, ResourceBudget
from research_repro.tools.execution import build_real_registry
from research_repro.tools.stub import build_stub_registry
from research_repro.experiments.fault_injection import FaultMode


def load_goals(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["goals"]


def make_goal(raw: dict[str, Any], config: Config) -> GoalContract:
    criterion = raw.get("criterion")
    budget = raw.get("budget") or {}
    max_exp = int(budget.get("max_experiments", config.max_experiments))
    max_rec = int(budget.get("max_total_recoveries", config.max_recoveries))
    max_sub = int(budget.get("max_recoveries_per_subtask", config.max_recoveries_per_subtask))
    return GoalContract(
        objective=raw["objective"],
        primary_metric=raw["primary_metric"],
        target_value=raw.get("target_value"),
        success_threshold=raw.get("success_threshold", 0.02),
        criterion=MetricCriterion(**criterion) if criterion else None,
        constraints=[Constraint(name=x, description=x) for x in raw.get("constraints", [])],
        max_experiments=max_exp,
        allowed_resources=ResourceBudget(
            max_experiments=max_exp,
            max_recoveries_per_subtask=max_sub,
            max_total_recoveries=max_rec,
        ),
    )


def environment_for(goal: dict[str, Any]) -> tuple[str, Path, Path]:
    name = "rag_retrieval" if any(x in goal["objective"].lower() for x in ("rag", "retrieval", "rerank", "ndcg", "mrr")) else "image_classification"
    env = ROOT / "environments" / name
    return name, env / ("eval_rag.py" if name == "rag_retrieval" else "train.py"), env / ("evaluate_rag.py" if name == "rag_retrieval" else "evaluate.py")


def parse_faults(raw: dict[str, Any]) -> dict[str, FaultMode]:
    faults: dict[str, FaultMode] = {}
    blob = raw.get("faults") or {}
    if raw.get("fault_mode") and not blob:
        blob = {"1": raw["fault_mode"]}
    for key, value in blob.items():
        if value in (None, "", "normal"):
            continue
        faults[str(key)] = FaultMode(str(value))
    return faults


def _count_events(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "tool_calls": 0,
        "self_corrections": 0,
        "recovery_attempts": 0,
        "recovery_failures": 0,
        "failures_detected": 0,
        "goal_drift_count": 0,
        "schema_error_count": 0,
        "timeout_count": 0,
        "duplicate_block_count": 0,
        "no_progress_count": 0,
        "experiments": 0,
    }
    for e in events:
        typ = e.get("event_type")
        payload = e.get("payload") or {}
        if typ == "tool_call":
            counts["tool_calls"] += 1
        elif typ == "experiment_completed":
            counts["experiments"] += 1
        elif typ == "failure_detected":
            counts["failures_detected"] += 1
            ft = str(payload.get("failure_type") or "")
            if ft == "goal_drift":
                counts["goal_drift_count"] += 1
            if ft == "tool_schema_error":
                counts["schema_error_count"] += 1
            if "timeout" in str(payload.get("description") or payload.get("reason") or "").lower() or ft == "tool_crash" and "timeout" in json.dumps(payload).lower():
                counts["timeout_count"] += 1
        elif typ == "recovery_started":
            counts["recovery_attempts"] += 1
            counts["self_corrections"] += 1
        elif typ == "recovery_completed":
            if not payload.get("success"):
                counts["recovery_failures"] += 1
        elif typ == "proposal_blocked":
            counts["duplicate_block_count"] += 1
        elif typ == "no_progress":
            counts["no_progress_count"] += 1
    return counts


def metrics_from_run(
    run_dir: Path,
    memory: dict[str, Any],
    report: dict[str, Any],
    agent_type: str = "self_correcting",
) -> dict[str, Any]:
    events_path = run_dir / "run.jsonl"
    events = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines() if x.strip()] if events_path.exists() else []
    counts = _count_events(events)
    best_id = memory.get("best_experiment_id")
    observed_id = memory.get("best_observed_accuracy_id")
    experiments = memory.get("experiments") or []
    best_valid = next((e for e in experiments if e.get("id") == best_id), None)
    best_obs = next((e for e in experiments if e.get("id") == observed_id), None)

    def acc_lat(exp: dict[str, Any] | None) -> tuple[float | None, float | None]:
        if not exp:
            return None, None
        result = exp.get("observed_result") or {}
        extra = result.get("additional_metrics") or {}
        return result.get("accuracy"), extra.get("latency_ms")

    valid_acc, valid_lat = acc_lat(best_valid)
    obs_acc, obs_lat = acc_lat(best_obs)
    last = experiments[-1] if experiments else None
    last_acc, last_lat = acc_lat(last)
    naive = agent_type == "naive"
    recoveries = memory.get("recoveries") or []
    by_strategy = {}
    for rec in recoveries:
        key = rec.get("strategy") or "unknown"
        by_strategy.setdefault(key, {"attempts": 0, "successes": 0})
        by_strategy[key]["attempts"] += 1
        if rec.get("success"):
            by_strategy[key]["successes"] += 1
    attempts = 0 if naive else counts["recovery_attempts"]
    failures = 0 if naive else (report.get("recovery_failures") or counts["recovery_failures"])
    return {
        "run_id": memory.get("run_id") or run_dir.name,
        "goal_id": None,
        "seed": None,
        "budget": None,
        "mode": agent_type,
        "completed": bool(report.get("completed")),
        "termination_reason": memory.get("terminate_reason"),
        "steps": report.get("total_steps", 0),
        "tool_calls": counts["tool_calls"],
        "experiments": counts["experiments"] or report.get("experiments_run", 0),
        "self_corrections": 0 if naive else (report.get("self_corrections") or counts["self_corrections"]),
        "retries": report.get("retries", 0) if naive else 0,
        "recovery_attempts": attempts,
        "recovery_failures": failures,
        "recovery_failure_rate": (failures / attempts) if attempts else 0,
        "failures_detected": counts["failures_detected"],
        "goal_drift_count": counts["goal_drift_count"],
        "schema_error_count": counts["schema_error_count"],
        "timeout_count": counts["timeout_count"],
        "duplicate_block_count": counts["duplicate_block_count"],
        "no_progress_count": memory.get("no_progress_count", counts["no_progress_count"]),
        "recovery_by_strategy": by_strategy if not naive else {},
        "best_valid_experiment": best_id,
        "best_valid_accuracy": valid_acc,
        "best_valid_latency": valid_lat,
        "best_observed_accuracy": obs_acc,
        "best_observed_accuracy_experiment": observed_id,
        "best_observed_latency": obs_lat,
        "final_accuracy": last_acc,
        "final_latency": last_lat,
        "independently_verified": (memory.get("reproduction") or {}).get("independently_verified"),
    }


def campaign_finished(run_dir: Path) -> bool:
    memory_path = run_dir / "memory.json"
    if not memory_path.exists():
        return False
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    phase = str(memory.get("current_phase") or "")
    if phase in {"completed", "terminated", "failed"} or memory.get("terminate_reason"):
        return True
    events_path = run_dir / "run.jsonl"
    if events_path.exists():
        try:
            text = events_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        if '"event_type": "final_report"' in text or '"event_type":"final_report"' in text:
            return True
    return False


def row_from_existing(
    raw: dict[str, Any],
    agent_type: str,
    output_root: Path,
    base_config: Config,
) -> dict[str, Any] | None:
    goal_id = raw["id"]
    run_dir = output_root / goal_id / agent_type
    memory_path = run_dir / "memory.json"
    if not memory_path.exists():
        return None
    memory = json.loads(memory_path.read_text(encoding="utf-8"))
    goal = make_goal(raw, base_config)
    faults = parse_faults(raw)
    environment, _, _ = environment_for(raw)
    report_dump: dict[str, Any] = {}
    events_path = run_dir / "run.jsonl"
    if events_path.exists():
        for line in reversed(events_path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("event_type") == "final_report":
                report_dump = (event.get("payload") or {}).get("report") or {}
                break
    if not report_dump:
        experiments = memory.get("experiments") or []
        report_dump = {
            "run_id": memory.get("run_id") or agent_type,
            "completed": memory.get("current_phase") == "completed",
            "total_steps": memory.get("step_count", 0),
            "experiments_run": len(experiments),
            "self_corrections": len(memory.get("recoveries") or []),
            "recovery_failures": 0,
            "retries": 0,
        }
    metrics = metrics_from_run(run_dir, memory, report_dump, agent_type)
    metrics["goal_id"] = goal_id
    metrics["seed"] = raw.get("seed", 42)
    metrics["budget"] = goal.allowed_resources.model_dump()
    return {
        "goal_id": goal_id,
        "description": raw.get("description"),
        "agent_type": agent_type,
        "run_dir": str(run_dir),
        "model": base_config.openai_model,
        "model_base_url": base_config.llm_base_url,
        "environment": environment,
        "timestamp": memory.get("created_at"),
        "seed": raw.get("seed", 42),
        "initial_configuration": raw.get("initial_configuration") or {},
        "fault_injection": {k: v.value if isinstance(v, FaultMode) else v for k, v in faults.items()},
        "budget": goal.allowed_resources.model_dump(),
        "report": report_dump,
        "metrics": metrics,
        "self_correcting": agent_type == "self_correcting",
        "fault": (list(faults.values())[0].value if faults else None),
        "memory": memory,
        "resumed": True,
    }


def run_one(
    raw: dict[str, Any],
    agent_type: str,
    output_root: Path,
    base_config: Config,
    stub_tools: bool = False,
) -> dict[str, Any]:
    goal_id = raw["id"]
    run_dir = output_root / goal_id / agent_type
    run_id = agent_type
    config = base_config.model_copy(update={"runs_dir": output_root / goal_id, "use_stub_tools": stub_tools})
    goal = make_goal(raw, config)
    environment, script, eval_script = environment_for(raw)
    faults = parse_faults(raw)
    if stub_tools:
        registry = build_stub_registry()
    else:
        registry = build_real_registry(
            run_dir / "experiments",
            script,
            eval_script,
            fault_schedule=faults or None,
            live_papers=False,
        )
    strategy = NaiveStrategy() if agent_type == "naive" else None
    started = datetime.now(timezone.utc).isoformat()
    report = AgentLoop(goal, run_id, config, registry, strategy=strategy).run()
    actual_run = run_dir / run_id if (run_dir / run_id / "memory.json").exists() else config.runs_dir / run_id
    memory_path = actual_run / "memory.json"
    memory = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.exists() else {}
    report_dump = report.model_dump(mode="json")
    metrics = metrics_from_run(actual_run, memory, report_dump, agent_type)
    metrics["goal_id"] = goal_id
    metrics["seed"] = raw.get("seed", 42)
    metrics["budget"] = goal.allowed_resources.model_dump()
    return {
        "goal_id": goal_id,
        "description": raw.get("description"),
        "agent_type": agent_type,
        "run_dir": str(actual_run),
        "model": config.openai_model,
        "model_base_url": config.llm_base_url,
        "environment": environment,
        "timestamp": started,
        "seed": raw.get("seed", 42),
        "initial_configuration": raw.get("initial_configuration") or {},
        "fault_injection": {k: v.value if isinstance(v, FaultMode) else v for k, v in faults.items()},
        "budget": goal.allowed_resources.model_dump(),
        "report": report_dump,
        "metrics": metrics,
        "self_correcting": agent_type == "self_correcting",
        "fault": (list(faults.values())[0].value if faults else None),
        "memory": memory,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for agent in ("self_correcting", "naive"):
        selected = [r for r in rows if r["agent_type"] == agent]
        mets = [r.get("metrics") or {} for r in selected]
        n = len(selected) or 1
        completed = sum(bool(m.get("completed")) for m in mets)
        recovery_fail_runs = sum(1 for m in mets if (m.get("recovery_attempts") or 0) > 0 and (m.get("recovery_failures") or 0) > 0)
        drift_ok = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "constraint_reject_and_rollback" and rec.get("success"))
        drift_n = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "constraint_reject_and_rollback")
        timeout_ok = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if "timeout" in str(rec.get("action_taken") or "").lower() and rec.get("success"))
        timeout_n = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if "timeout" in str(rec.get("action_taken") or "").lower() or rec.get("strategy") == "tool_crash_repair_and_retry")
        schema_ok = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "schema_validate_and_repair" and rec.get("success"))
        schema_n = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "schema_validate_and_repair")
        inconsistency_ok = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "independent_metric_reaudit" and rec.get("success"))
        inconsistency_n = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "independent_metric_reaudit")
        regression_ok = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "best_node_rollback_and_prune" and rec.get("success"))
        regression_n = sum(1 for r in selected for rec in (r.get("memory") or {}).get("recoveries", []) if rec.get("strategy") == "best_node_rollback_and_prune")
        rec_attempts = sum(m.get("recovery_attempts") or 0 for m in mets)
        rec_fails = sum(m.get("recovery_failures") or 0 for m in mets)
        out[agent] = {
            "runs": len(selected),
            "completed_goals": completed,
            "completion_rate": completed / len(selected) if selected else 0.0,
            "completion_fraction": f"{completed}/{len(selected)}" if selected else "0/0",
            "mean_steps_all_runs": sum(m.get("steps") or 0 for m in mets) / n,
            "mean_steps_completed_runs": (
                sum(m.get("steps") or 0 for m in mets if m.get("completed")) / completed
                if completed else None
            ),
            "total_self_corrections": sum(m.get("self_corrections") or 0 for m in mets),
            "mean_self_corrections": sum(m.get("self_corrections") or 0 for m in mets) / n,
            "total_retries": sum(m.get("retries") or 0 for m in mets),
            "mean_retries": sum(m.get("retries") or 0 for m in mets) / n,
            "recovery_attempts": rec_attempts,
            "recovery_failures": rec_fails,
            "recovery_failure_rate": (rec_fails / rec_attempts) if rec_attempts else 0,
            "recovery_failure_runs": recovery_fail_runs,
            "goal_drift_recoveries": drift_n,
            "goal_drift_recovery_success": (drift_ok / drift_n) if drift_n else None,
            "timeout_recoveries": timeout_n,
            "timeout_recovery_success": (timeout_ok / timeout_n) if timeout_n else None,
            "schema_recoveries": schema_n,
            "schema_recovery_success": (schema_ok / schema_n) if schema_n else None,
            "result_inconsistency_recoveries": inconsistency_n,
            "result_inconsistency_recovery_success": (inconsistency_ok / inconsistency_n) if inconsistency_n else None,
            "regression_recoveries": regression_n,
            "regression_recovery_success": (regression_ok / regression_n) if regression_n else None,
        }
    per_goal = []
    ids = sorted({r["goal_id"] for r in rows})
    by = {(r["goal_id"], r["agent_type"]): r for r in rows}
    for gid in ids:
        sc = by.get((gid, "self_correcting"), {}).get("metrics") or {}
        nv = by.get((gid, "naive"), {}).get("metrics") or {}
        per_goal.append({
            "goal_id": gid,
            "self_correcting": "PASS" if sc.get("completed") else "FAIL",
            "naive": "PASS" if nv.get("completed") else "FAIL",
            "self_steps": sc.get("steps"),
            "naive_steps": nv.get("steps"),
            "self_reason": (
                "completed" if sc.get("completed")
                else (sc.get("termination_reason") or "incomplete")
            ),
            "naive_reason": (
                "completed" if nv.get("completed")
                else (nv.get("termination_reason") or "incomplete")
            ),
            "self_correcting_metrics": sc,
            "naive_metrics": nv,
        })
    out["per_goal"] = per_goal
    return out


def extract_traces(rows: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    traces = []
    for r in rows:
        if r.get("agent_type") != "self_correcting":
            continue
        memory = r.get("memory") or {}
        experiments = memory.get("experiments") or []
        failures = memory.get("failures") or []
        recoveries = memory.get("recoveries") or []
        if not failures and not recoveries:
            continue
        traces.append({
            "goal_id": r["goal_id"],
            "run_dir": r["run_dir"],
            "initial_state": experiments[0] if experiments else None,
            "failures": failures,
            "recoveries": recoveries,
            "hypotheses": [h.get("statement") for h in memory.get("hypotheses") or []],
            "experiments": [
                {
                    "id": e.get("id"),
                    "parent_id": e.get("parent_id"),
                    "hypothesis": e.get("hypothesis"),
                    "parameters": e.get("parameters"),
                    "accuracy": (e.get("observed_result") or {}).get("accuracy"),
                    "latency_ms": ((e.get("observed_result") or {}).get("additional_metrics") or {}).get("latency_ms"),
                    "outcome": None,
                }
                for e in experiments
            ],
            "final_outcome": r.get("metrics"),
        })
        if len(traces) >= limit:
            break
    return traces


def _write_result(path: Path, rows: list[dict[str, Any]], stub_tools: bool) -> None:
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stub_tools": stub_tools,
        "execution_mode": "stub" if stub_tools else "real",
        "denominator_note": "average_steps = total steps across all runs / total runs (failed runs included)",
        "runs": rows,
        "summary": summarize(rows),
        "correction_traces": extract_traces(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", type=Path, default=Path(__file__).with_name("goals.yaml"))
    parser.add_argument("--output", type=Path, default=ROOT / "evaluations" / "results")
    parser.add_argument("--goal", action="append")
    parser.add_argument("--stub-tools", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Skip campaigns that already wrote a final report")
    args = parser.parse_args()
    config = Config.from_env()
    goals = [g for g in load_goals(args.goals) if not args.goal or g["id"] in args.goal]
    rows: list[dict[str, Any]] = []
    dest = args.output / "benchmark_results.json"
    for goal in goals:
        for agent in ("self_correcting", "naive"):
            run_dir = args.output / goal["id"] / agent
            if args.resume and campaign_finished(run_dir):
                existing = row_from_existing(goal, agent, args.output, config)
                if existing:
                    rows.append(existing)
                    print(f"RESUME skip {goal['id']}/{agent} completed={existing['metrics'].get('completed')}")
                    continue
            try:
                rows.append(run_one(goal, agent, args.output, config, stub_tools=args.stub_tools))
            except Exception as exc:
                print(f"ERROR {goal['id']}/{agent}: {type(exc).__name__}: {exc}")
                rows.append({
                    "goal_id": goal["id"],
                    "agent_type": agent,
                    "run_dir": str(run_dir),
                    "error": f"{type(exc).__name__}: {exc}",
                    "metrics": {
                        "run_id": agent,
                        "goal_id": goal["id"],
                        "mode": agent,
                        "completed": False,
                        "termination_reason": f"runner_exception:{type(exc).__name__}",
                        "steps": 0,
                        "self_corrections": 0,
                        "retries": 0,
                        "recovery_attempts": 0,
                        "recovery_failures": 0,
                    },
                    "memory": {},
                    "self_correcting": agent == "self_correcting",
                    "seed": goal.get("seed", 42),
                    "budget": (goal.get("budget") or {}),
                    "fault": None,
                })
            _write_result(dest, rows, args.stub_tools)
    _write_result(dest, rows, args.stub_tools)
    print(json.dumps(json.loads(dest.read_text(encoding="utf-8"))["summary"], indent=2, default=str))


if __name__ == "__main__":
    main()
