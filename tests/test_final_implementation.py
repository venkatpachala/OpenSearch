"""Tests for FINAL_IMPLEMENTATION.md hardening and benchmark aggregation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))

from research_repro.agent.planning_context import (
    assess_proposal,
    configuration_fingerprint,
    JUSTIFIED_RETRY_REASONS,
)
from research_repro.evaluation.evaluator import SelfEvaluator
from research_repro.experiments.fault_injection import FaultInjectingRunner, FaultMode
from research_repro.experiments.runner import ExperimentConfig, ExperimentRunner
from research_repro.memory.models import (
    ConfigurationOutcome,
    ConfigurationRecord,
    Constraint,
    Failure,
    FailureType,
    GoalContract,
    MetricCriterion,
    ResearchMemory,
)
from research_repro.observability.benchmark_report import render_markdown
from evaluations.run_benchmark import load_goals, summarize


def _goal() -> GoalContract:
    return GoalContract(
        objective="Find highest possible MNIST accuracy keeping latency below 100 ms",
        primary_metric="accuracy",
        target_value=0.95,
        criterion=MetricCriterion(metric="accuracy", direction="maximize", minimum=0.95, target=0.95, tolerance=0.0),
        constraints=[Constraint(name="latency_under_100ms", description="latency_under_100ms")],
    )


def test_allocate_experiment_ids_are_unique_and_ignore_reuse(tmp_path):
    runner = ExperimentRunner(tmp_path / "experiments")
    first = runner.allocate_experiment_id("experiment_001")
    (tmp_path / "experiments" / first).mkdir(parents=True, exist_ok=True)
    (tmp_path / "experiments" / first / "metrics.json").write_text("{}", encoding="utf-8")
    second = runner.allocate_experiment_id("experiment_001")
    assert first != second
    assert first.startswith("experiment_")
    assert second.startswith("experiment_")


def test_runner_refuses_to_overwrite_completed_experiment(tmp_path):
    runner = ExperimentRunner(tmp_path / "experiments")
    script = tmp_path / "noop.py"
    script.write_text(
        "import argparse\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser()\np.add_argument('--output-dir')\n"
        "p.add_argument('--seed', type=int, default=42)\nargs=p.parse_args()\n"
        "Path(args.output_dir).mkdir(parents=True, exist_ok=True)\n"
        "(Path(args.output_dir)/'metrics.json').write_text('{\"accuracy\":0.9}')\n"
    )
    eid = "experiment_001"
    out = runner.experiment_dir(eid)
    cfg = ExperimentConfig(experiment_id=eid, script_path=script, output_dir=out)
    first = runner.run(cfg)
    assert first.success
    try:
        runner.run(cfg)
        assert False, "overwrite should be refused"
    except FileExistsError:
        pass


def test_canonical_config_is_written(tmp_path):
    runner = ExperimentRunner(tmp_path / "experiments")
    script = tmp_path / "noop.py"
    script.write_text(
        "import argparse\nfrom pathlib import Path\n"
        "p=argparse.ArgumentParser()\np.add_argument('--output-dir')\n"
        "p.add_argument('--seed', type=int, default=42)\n"
        "p.add_argument('--lr', type=float, default=0.001)\n"
        "args=p.parse_args()\nPath(args.output_dir).mkdir(parents=True, exist_ok=True)\n"
        "(Path(args.output_dir)/'metrics.json').write_text('{\"accuracy\":0.9}')\n"
    )
    out = runner.experiment_dir("experiment_001")
    runner.run(ExperimentConfig(
        experiment_id="experiment_001",
        script_path=script,
        output_dir=out,
        parameters={"learning_rate": 0.001, "hidden_size": 64},
    ))
    saved = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert saved["parameters"]["lr"] == 0.001
    assert saved["parameters"]["hidden_size"] == 64
    assert "learning_rate" not in saved["parameters"]
    assert (out / "stdout.log").exists()
    assert (out / "stderr.log").exists()


def test_fingerprint_order_and_aliases_match_defaults():
    a = configuration_fingerprint({"parameters": {"lr": 0.001, "hidden_size": 128}})
    b = configuration_fingerprint({"parameters": {"hidden_size": 128, "lr": 0.001}})
    c = configuration_fingerprint({"parameters": {"learning_rate": 0.001}})
    d = configuration_fingerprint({"parameters": {}})
    assert a == b == c == d


def test_justified_retry_is_allowed():
    mem = ResearchMemory(goal=_goal(), run_id="retry")
    fp = configuration_fingerprint({"parameters": {"hidden_size": 64, "hidden_layers": 1}})
    mem.tested_configs[fp] = ConfigurationRecord(
        fingerprint=fp,
        config={"hidden_size": 64, "hidden_layers": 1},
        experiment_id="experiment_001",
        accuracy=0.91,
        latency_ms=76.1,
        outcome=ConfigurationOutcome.TOOL_FAILURE,
    )
    blocked = assess_proposal(
        mem, "run_experiment",
        {"experiment_id": "x", "parameters": {"hidden_size": 64, "hidden_layers": 1}},
    )
    assert blocked.blocked is True
    allowed = assess_proposal(
        mem, "run_experiment",
        {
            "experiment_id": "x",
            "parameters": {"hidden_size": 64, "hidden_layers": 1},
            "retry": True,
            "retry_reason": "transient_timeout",
        },
    )
    assert allowed.blocked is False
    assert "transient_timeout" in JUSTIFIED_RETRY_REASONS


def test_fault_ordinal_matches_first_allocated_id(tmp_path):
    base = ExperimentRunner(tmp_path / "experiments")
    wrapped = FaultInjectingRunner(base, fault_schedule={"1": FaultMode.TIMEOUT, "exp_01": FaultMode.TIMEOUT})
    script = tmp_path / "noop.py"
    script.write_text("import argparse\nargparse.ArgumentParser().parse_args([])\n")
    eid = base.allocate_experiment_id()
    cfg = ExperimentConfig(experiment_id=eid, script_path=script, output_dir=base.experiment_dir(eid), timeout_seconds=5)
    result = wrapped.run(cfg)
    assert result.timed_out is True


def test_goals_yaml_has_ten_distinct_goals():
    goals = load_goals(Path(__file__).resolve().parents[1] / "evaluations" / "goals.yaml")
    assert len(goals) >= 10
    ids = [g["id"] for g in goals]
    assert len(set(ids)) == len(ids)
    assert {g["id"] for g in goals} >= {f"G{i:02d}" for i in range(1, 11)}
    descriptions = [g.get("description") for g in goals]
    assert len(set(descriptions)) == len(descriptions)


def test_campaign_finished_detects_final_report_and_completed_phase(tmp_path):
    from evaluations.run_benchmark import campaign_finished
    run_dir = tmp_path / "G01" / "self_correcting"
    run_dir.mkdir(parents=True)
    (run_dir / "memory.json").write_text('{"current_phase": "experiment_planning"}', encoding="utf-8")
    assert campaign_finished(run_dir) is False
    (run_dir / "run.jsonl").write_text(
        '{"event_type":"final_report","payload":{"report":{"completed":false}}}\n',
        encoding="utf-8",
    )
    assert campaign_finished(run_dir) is True
    run_dir2 = tmp_path / "G02" / "naive"
    run_dir2.mkdir(parents=True)
    (run_dir2 / "memory.json").write_text(
        '{"current_phase": "self_evaluation", "terminate_reason": "no_progress"}',
        encoding="utf-8",
    )
    assert campaign_finished(run_dir2) is True


def test_benchmark_summary_aggregates_completion_rates():
    rows = [
        {"goal_id": "G01", "agent_type": "self_correcting", "metrics": {"completed": True, "steps": 4, "self_corrections": 2, "recovery_attempts": 2, "recovery_failures": 0}, "memory": {"recoveries": []}},
        {"goal_id": "G01", "agent_type": "naive", "metrics": {"completed": False, "steps": 3, "self_corrections": 0, "recovery_attempts": 0, "recovery_failures": 0}, "memory": {"recoveries": []}},
        {"goal_id": "G02", "agent_type": "self_correcting", "metrics": {"completed": False, "steps": 5, "self_corrections": 1, "recovery_attempts": 1, "recovery_failures": 1}, "memory": {"recoveries": []}},
        {"goal_id": "G02", "agent_type": "naive", "metrics": {"completed": False, "steps": 5, "self_corrections": 0, "recovery_attempts": 0, "recovery_failures": 0}, "memory": {"recoveries": []}},
    ]
    summary = summarize(rows)
    assert summary["self_correcting"]["completed_goals"] == 1
    assert summary["self_correcting"]["completion_rate"] == 0.5
    assert summary["naive"]["completion_rate"] == 0.0
    assert summary["naive"]["mean_self_corrections"] == 0
    assert summary["self_correcting"]["recovery_failure_runs"] == 1
    assert summary["per_goal"][0]["self_correcting"] == "PASS"
    assert summary["per_goal"][0]["naive"] == "FAIL"
    md = render_markdown({"summary": summary, "correction_traces": [], "denominator_note": "all runs"})
    assert "Completion" in md
    assert "G01" in md
    assert "Naive retries are **not** counted as self-corrections." in md


def test_independent_eval_has_no_noise_source():
    src = (Path(__file__).resolve().parents[1] / "environments" / "image_classification" / "evaluate.py").read_text(encoding="utf-8")
    assert "RandomState" not in src
    assert "normal(0" not in src


def test_production_train_has_no_silent_synthetic_fallback():
    src = (Path(__file__).resolve().parents[1] / "environments" / "image_classification" / "train.py").read_text(encoding="utf-8")
    assert "--synthetic" in src
    assert "refusing to substitute synthetic data" in src.lower() or "mnist unavailable" in src.lower()
    assert "if synthetic" in src


def test_latency_is_documented_as_cost_proxy():
    src = (Path(__file__).resolve().parents[1] / "environments" / "image_classification" / "train.py").read_text(encoding="utf-8")
    assert "not measured serving latency" in src.lower() or "cost proxy" in src.lower()


def test_stub_independent_eval_is_null_not_claimed():
    from research_repro.tools.stub import RunIndependentEvalRequest, StubIndependentEvalTool
    resp = StubIndependentEvalTool().execute(RunIndependentEvalRequest(experiment_id="experiment_001"))
    assert resp.data.get("independently_verified") is None
    assert resp.data.get("eval_accuracy") is None


def test_naive_metrics_zero_self_corrections_and_record_retries(tmp_path):
    from evaluations.run_benchmark import metrics_from_run
    run_dir = tmp_path / "naive"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(
        '{"event_type":"failure_detected","payload":{"failure_type":"goal_drift"}}\n'
        '{"event_type":"recovery_started","payload":{}}\n'
        '{"event_type":"recovery_completed","payload":{"success":false}}\n',
        encoding="utf-8",
    )
    memory = {
        "run_id": "naive",
        "best_experiment_id": None,
        "best_observed_accuracy_id": None,
        "experiments": [],
        "recoveries": [],
        "reproduction": {"independently_verified": None},
    }
    report = {"completed": False, "total_steps": 3, "self_corrections": 2, "recovery_failures": 1, "retries": 2, "experiments_run": 1}
    mets = metrics_from_run(run_dir, memory, report, agent_type="naive")
    assert mets["self_corrections"] == 0
    assert mets["retries"] == 2
    assert mets["recovery_attempts"] == 0
    assert mets["recovery_failures"] == 0
    assert mets["independently_verified"] is None


def test_self_correcting_metrics_keep_recoveries(tmp_path):
    from evaluations.run_benchmark import metrics_from_run
    run_dir = tmp_path / "self_correcting"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(
        '{"event_type":"recovery_started","payload":{}}\n'
        '{"event_type":"recovery_completed","payload":{"success":true}}\n',
        encoding="utf-8",
    )
    memory = {
        "run_id": "self_correcting",
        "best_experiment_id": "experiment_002",
        "best_observed_accuracy_id": "experiment_001",
        "experiments": [
            {"id": "experiment_001", "observed_result": {"accuracy": 0.961, "additional_metrics": {"latency_ms": 156.9}}},
            {"id": "experiment_002", "observed_result": {"accuracy": 0.9575, "additional_metrics": {"latency_ms": 76.1}}},
        ],
        "recoveries": [{"strategy": "constraint_reject_and_rollback", "success": True}],
        "reproduction": {"independently_verified": True},
    }
    report = {"completed": True, "total_steps": 5, "self_corrections": 2, "recovery_failures": 0, "retries": 0, "experiments_run": 2}
    mets = metrics_from_run(run_dir, memory, report, agent_type="self_correcting")
    assert mets["self_corrections"] == 2
    assert mets["retries"] == 0
    assert mets["recovery_attempts"] == 1
    assert mets["best_valid_experiment"] == "experiment_002"
    assert mets["best_observed_accuracy"] == 0.961
    assert mets["independently_verified"] is True
