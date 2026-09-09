"""Run the real self-correcting vs naive benchmark.

This module intentionally has no expected outcomes. It launches the same
AgentLoop, tools, evaluator, environments, and model twice per goal and
derives the scorecard from persisted memory/JSONL artifacts.
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
from research_repro.experiments.fault_injection import FaultMode


def load_goals(path: Path) -> list[dict[str, Any]]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["goals"]


def make_goal(raw: dict[str, Any], config: Config) -> GoalContract:
    criterion = raw.get("criterion")
    return GoalContract(
        objective=raw["objective"], primary_metric=raw["primary_metric"],
        target_value=raw.get("target_value"), success_threshold=raw.get("success_threshold", .02),
        criterion=MetricCriterion(**criterion) if criterion else None,
        constraints=[Constraint(name=x, description=x) for x in raw.get("constraints", [])],
        max_experiments=config.max_experiments,
        allowed_resources=ResourceBudget(
            max_experiments=config.max_experiments,
            max_recoveries_per_subtask=config.max_recoveries_per_subtask,
            max_total_recoveries=config.max_recoveries,
        ),
    )


def environment_for(goal: dict[str, Any]) -> tuple[str, Path, Path]:
    name = "rag_retrieval" if any(x in goal["objective"].lower() for x in ("rag", "retrieval", "rerank", "ndcg", "mrr")) else "image_classification"
    env = ROOT / "environments" / name
    return name, env / ("eval_rag.py" if name == "rag_retrieval" else "train.py"), env / ("evaluate_rag.py" if name == "rag_retrieval" else "evaluate.py")


def run_one(raw: dict[str, Any], agent_type: str, output_root: Path, base_config: Config) -> dict[str, Any]:
    goal_id = raw["id"]
    run_dir = output_root / goal_id / agent_type
    # AgentLoop appends run_id to runs_dir; use the agent type as the run id
    # so the persisted layout is exactly Gxx/{self_correcting,naive}/.
    run_id = agent_type
    config = base_config.model_copy(update={"runs_dir": output_root / goal_id})
    goal = make_goal(raw, config)
    environment, script, eval_script = environment_for(raw)
    fault = FaultMode(raw["fault_mode"]) if raw.get("fault_mode") else None
    registry = build_real_registry(
        run_dir / "experiments", script, eval_script,
        live_papers=True, default_fault_mode=fault,
    )
    strategy = NaiveStrategy() if agent_type == "naive" else None
    started = datetime.now(timezone.utc).isoformat()
    report = AgentLoop(goal, run_id, config, registry, strategy=strategy).run()
    memory_path = run_dir / run_id / "memory.json"
    actual_run = run_dir / run_id
    return {
        "goal_id": goal_id, "agent_type": agent_type, "run_dir": str(actual_run),
        "model": config.openai_model, "model_base_url": config.llm_base_url,
        "environment": environment, "timestamp": started,
        "report": report.model_dump(mode="json"),
        "memory": json.loads(memory_path.read_text()) if memory_path.exists() else {},
        "fault_mode": raw.get("fault_mode"), "budget": goal.allowed_resources.model_dump(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for agent in ("self_correcting", "naive"):
        selected = [r for r in rows if r["agent_type"] == agent]
        reports = [r["report"] for r in selected]
        event_counts = {"recoverable_failures": 0, "successful_recoveries": 0, "experiments": 0, "budget_exhausted": 0}
        failure_types: dict[str, dict[str, int]] = {}
        for r in selected:
            events_path = Path(r["run_dir"]) / "run.jsonl"
            events = [json.loads(x) for x in events_path.read_text(encoding="utf-8").splitlines()] if events_path.exists() else []
            for e in events:
                typ, payload = e.get("event_type"), e.get("payload", {})
                if typ == "failure_detected" and payload.get("failure_type"):
                    ft = payload["failure_type"]; failure_types.setdefault(ft, {"detected": 0, "recovered": 0}); failure_types[ft]["detected"] += 1
                if typ == "recovery_completed": event_counts["successful_recoveries"] += int(bool(payload.get("success")))
                if typ == "experiment_started": event_counts["experiments"] += 1
                if typ == "budget_exhausted": event_counts["budget_exhausted"] += 1
            for f in r["memory"].get("failures", []):
                if f.get("recovered") and f.get("failure_type") in failure_types: failure_types[f["failure_type"]]["recovered"] += 1
            event_counts["recoverable_failures"] += len(r["memory"].get("failures", []))
        out[agent] = {
            "completion_rate": sum(bool(x.get("completed")) for x in reports) / len(reports) if reports else 0,
            "mean_steps": sum(x.get("total_steps", 0) for x in reports) / len(reports) if reports else 0,
            "mean_experiments": event_counts["experiments"] / len(reports) if reports else 0,
            "budget_exhaustion_rate": event_counts["budget_exhausted"] / len(reports) if reports else 0,
            "recovery_success_rate": event_counts["successful_recoveries"] / event_counts["recoverable_failures"] if event_counts["recoverable_failures"] else 0,
            "failure_types": failure_types,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", type=Path, default=Path(__file__).with_name("goals.yaml"))
    parser.add_argument("--output", type=Path, default=ROOT / "evaluations" / "results")
    parser.add_argument("--goal", action="append")
    args = parser.parse_args()
    config = Config.from_env()
    goals = [g for g in load_goals(args.goals) if not args.goal or g["id"] in args.goal]
    rows = []
    for goal in goals:
        for agent in ("self_correcting", "naive"):
            rows.append(run_one(goal, agent, args.output, config))
    result = {"generated_at": datetime.now(timezone.utc).isoformat(), "single_run_warning": "Results are indicative when repetitions=1", "runs": rows, "summary": summarize(rows)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "benchmark_results.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__": main()
