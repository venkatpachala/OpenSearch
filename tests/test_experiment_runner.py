"""
Tests for Phase 4: ExperimentRunner, ArtifactCollector, FaultInjectingRunner.
"""
import json
import sys
import tempfile
from pathlib import Path

import pytest

from research_repro.experiments.runner import (
    ExperimentConfig, ExperimentRunner, ExecutionResult,
)
from research_repro.experiments.artifacts import ArtifactCollector
from research_repro.experiments.fault_injection import (
    FaultInjectingRunner, FaultMode,
)
from research_repro.experiments.checkpoints import GitCheckpointer


# ---------------------------------------------------------------------------
# ExperimentRunner
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_creates_experiment_dir(self, tmp_path):
        runner = ExperimentRunner(base_dir=tmp_path / "experiments")
        exp_dir = runner.experiment_dir("exp_001")
        assert exp_dir.exists()

    def test_run_simple_python_script(self, tmp_path):
        """Run a trivial Python script that writes metrics.json."""
        script = tmp_path / "trivial_train.py"
        script.write_text(
            'import json, sys\n'
            'from pathlib import Path\n'
            'import argparse\n'
            'p = argparse.ArgumentParser()\n'
            'p.add_argument("--output-dir")\n'
            'p.add_argument("--seed", type=int, default=42)\n'
            'args = p.parse_args()\n'
            'out = Path(args.output_dir)\n'
            'out.mkdir(parents=True, exist_ok=True)\n'
            '(out / "metrics.json").write_text(json.dumps({"accuracy": 0.91, "f1": 0.89}))\n'
            'print("done")\n'
        )
        runner = ExperimentRunner(base_dir=tmp_path / "experiments")
        config = ExperimentConfig(
            experiment_id="exp_001",
            script_path=script,
            output_dir=runner.experiment_dir("exp_001"),
            parameters={},
            seed=42,
        )
        result = runner.run(config)
        assert result.success
        assert result.exit_code == 0
        assert not result.timed_out

    def test_run_failing_script(self, tmp_path):
        """Script that exits with code 1."""
        script = tmp_path / "fail_train.py"
        script.write_text('import sys\nprint("ERROR", file=sys.stderr)\nsys.exit(1)\n')
        runner = ExperimentRunner(base_dir=tmp_path / "experiments")
        config = ExperimentConfig(
            experiment_id="exp_fail",
            script_path=script,
            output_dir=runner.experiment_dir("exp_fail"),
            seed=42,
        )
        result = runner.run(config)
        assert not result.success
        assert result.exit_code == 1

    def test_config_written_to_disk(self, tmp_path):
        """Confirm config.json is written for reproducibility."""
        script = tmp_path / "noop.py"
        script.write_text('import argparse\np=argparse.ArgumentParser()\np.add_argument("--output-dir")\np.add_argument("--seed",type=int,default=42)\np.parse_args()\n')
        runner = ExperimentRunner(base_dir=tmp_path / "experiments")
        config = ExperimentConfig(
            experiment_id="exp_cfg",
            script_path=script,
            output_dir=runner.experiment_dir("exp_cfg"),
            parameters={"normalize": True},
            seed=99,
        )
        runner.run(config)
        cfg_file = runner.experiment_dir("exp_cfg") / "config.json"
        assert cfg_file.exists()
        data = json.loads(cfg_file.read_text())
        assert data["seed"] == 99

    def test_collect_metrics_from_json(self, tmp_path):
        output_dir = tmp_path / "exp_metrics"
        output_dir.mkdir()
        (output_dir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.934, "f1": 0.929, "loss": 0.21})
        )
        runner = ExperimentRunner(base_dir=tmp_path)
        result = ExecutionResult(
            experiment_id="exp_metrics",
            exit_code=0,
            stdout="",
            stderr="",
            runtime_seconds=1.0,
            output_dir=output_dir,
        )
        metrics = runner.collect_metrics(result)
        assert metrics["accuracy"] == pytest.approx(0.934)

    def test_collect_metrics_missing(self, tmp_path):
        runner = ExperimentRunner(base_dir=tmp_path)
        result = ExecutionResult(
            experiment_id="no_metrics",
            exit_code=0,
            stdout="",
            stderr="",
            runtime_seconds=1.0,
            output_dir=tmp_path / "nonexistent",
        )
        metrics = runner.collect_metrics(result)
        assert metrics == {}


# ---------------------------------------------------------------------------
# ArtifactCollector
# ---------------------------------------------------------------------------


class TestArtifactCollector:
    def test_collect_from_empty_dir(self, tmp_path):
        collector = ArtifactCollector()
        result = collector.collect("exp_001", tmp_path / "nonexistent")
        assert not result.has_metrics
        assert result.artifacts == []

    def test_collect_metrics_json(self, tmp_path):
        output_dir = tmp_path / "exp_001"
        output_dir.mkdir()
        (output_dir / "metrics.json").write_text(
            json.dumps({"accuracy": 0.91, "f1": 0.89, "loss": 0.35})
        )
        collector = ArtifactCollector()
        result = collector.collect("exp_001", output_dir)
        assert result.has_metrics
        assert result.get_metric("accuracy") == pytest.approx(0.91)
        assert result.get_metric("f1") == pytest.approx(0.89)

    def test_collect_classifies_files(self, tmp_path):
        output_dir = tmp_path / "exp_002"
        output_dir.mkdir()
        (output_dir / "metrics.json").write_text("{}")
        (output_dir / "config.json").write_text("{}")
        (output_dir / "run.log").write_text("logs")

        collector = ArtifactCollector()
        result = collector.collect("exp_002", output_dir)
        types = {a.name: a.artifact_type for a in result.artifacts}
        assert types["metrics.json"] == "metrics"
        assert types["config.json"] == "config"
        assert types["run.log"] == "log"

    def test_summary_structure(self, tmp_path):
        output_dir = tmp_path / "exp_sum"
        output_dir.mkdir()
        (output_dir / "metrics.json").write_text(json.dumps({"accuracy": 0.92}))
        collector = ArtifactCollector()
        result = collector.collect("exp_sum", output_dir)
        summary = result.summary()
        assert summary["experiment_id"] == "exp_sum"
        assert summary["artifact_count"] >= 1
        assert "accuracy" in summary["metrics"]


# ---------------------------------------------------------------------------
# FaultInjectingRunner
# ---------------------------------------------------------------------------


class TestFaultInjectingRunner:
    def _make_base_runner(self, tmp_path: Path) -> ExperimentRunner:
        return ExperimentRunner(base_dir=tmp_path / "experiments")

    def _make_config(self, runner: ExperimentRunner, exp_id: str, tmp_path: Path) -> ExperimentConfig:
        script = tmp_path / "mock_train.py"
        if not script.exists():
            script.write_text(
                'import json, argparse\n'
                'from pathlib import Path\n'
                'p = argparse.ArgumentParser()\n'
                'p.add_argument("--output-dir")\n'
                'p.add_argument("--seed", type=int, default=42)\n'
                'args = p.parse_args()\n'
                'out = Path(args.output_dir)\n'
                'out.mkdir(parents=True, exist_ok=True)\n'
                '(out / "metrics.json").write_text(json.dumps({"accuracy": 0.91}))\n'
            )
        return ExperimentConfig(
            experiment_id=exp_id,
            script_path=script,
            output_dir=runner.experiment_dir(exp_id),
            seed=42,
        )

    def test_fail_once_fails_first_succeeds_second(self, tmp_path):
        base = self._make_base_runner(tmp_path)
        fault_runner = FaultInjectingRunner(
            base_runner=base,
            fault_schedule={"exp_f1": FaultMode.FAIL_ONCE},
        )
        config = self._make_config(base, "exp_f1", tmp_path)

        first = fault_runner.run(config)
        assert not first.success, "First attempt should fail"
        assert first.exit_code == 1

        second = fault_runner.run(config)
        assert second.success, "Second attempt should succeed"

    def test_timeout_returns_timed_out_result(self, tmp_path):
        base = self._make_base_runner(tmp_path)
        fault_runner = FaultInjectingRunner(
            base_runner=base,
            fault_schedule={"exp_to": FaultMode.TIMEOUT},
        )
        config = self._make_config(base, "exp_to", tmp_path)
        config.timeout_seconds = 30

        result = fault_runner.run(config)
        assert result.timed_out
        assert not result.success

    def test_missing_metrics_exits_zero_no_metrics(self, tmp_path):
        base = self._make_base_runner(tmp_path)
        fault_runner = FaultInjectingRunner(
            base_runner=base,
            fault_schedule={"exp_mm": FaultMode.MISSING_METRICS},
        )
        config = self._make_config(base, "exp_mm", tmp_path)
        result = fault_runner.run(config)
        assert result.exit_code == 0
        assert not result.timed_out
        # metrics.json should NOT exist
        assert not (result.output_dir / "metrics.json").exists()

    def test_wrong_result_writes_latency_violation(self, tmp_path):
        base = self._make_base_runner(tmp_path)
        fault_runner = FaultInjectingRunner(
            base_runner=base,
            fault_schedule={"exp_wr": FaultMode.WRONG_RESULT},
        )
        config = self._make_config(base, "exp_wr", tmp_path)
        result = fault_runner.run(config)
        assert result.success
        metrics_path = result.output_dir / "metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text())
        assert metrics["latency_ms"] > 100  # Violates constraint
        assert metrics["accuracy"] > 0.95  # But good accuracy

    def test_normal_mode_is_passthrough(self, tmp_path):
        base = self._make_base_runner(tmp_path)
        fault_runner = FaultInjectingRunner(
            base_runner=base,
            default_mode=FaultMode.NORMAL,
        )
        config = self._make_config(base, "exp_normal", tmp_path)
        result = fault_runner.run(config)
        assert result.success


# ---------------------------------------------------------------------------
# GitCheckpointer
# ---------------------------------------------------------------------------


class TestGitCheckpointer:
    def test_initialize_repo(self, tmp_path):
        checkpointer = GitCheckpointer(workspace_dir=tmp_path)
        ok = checkpointer.initialize()
        assert ok is True
        assert (tmp_path / ".git").exists()

    def test_checkpoint_creates_commit(self, tmp_path):
        checkpointer = GitCheckpointer(workspace_dir=tmp_path)
        checkpointer.initialize()

        test_file = tmp_path / "model.txt"
        test_file.write_text("model weights v1")

        commit_hash = checkpointer.checkpoint("exp_001", "first model")
        assert commit_hash is not None
        assert len(commit_hash) >= 7

        log = checkpointer.log(n=5)
        assert any("exp_001" in entry["message"] for entry in log)

    def test_restore_checkpoint(self, tmp_path):
        checkpointer = GitCheckpointer(workspace_dir=tmp_path)
        checkpointer.initialize()

        test_file = tmp_path / "config.json"
        test_file.write_text('{"lr": 0.01}')
        c1 = checkpointer.checkpoint("exp_001", "good config")

        # Mutate to bad config
        test_file.write_text('{"lr": 999.0}')
        c2 = checkpointer.checkpoint("exp_002", "bad config")

        # Restore to c1
        restored = checkpointer.restore(c1)
        assert restored is True
        assert test_file.read_text() == '{"lr": 0.01}'
