"""
Fault Injection System — controlled experiment failures for demo and testing.

Wraps ExperimentRunner with configurable failure modes to reliably demonstrate
the agent's self-correction capabilities without relying on real failures.

Failure Modes:
  NORMAL          — pass-through, no injection
  FAIL_ONCE       — crash on first call for an experiment_id, succeed after
  TIMEOUT         — simulate a process timeout
  CORRUPT_METRIC  — return plausible but wrong accuracy
  WRONG_RESULT    — return result that violates a constraint (goal drift demo)
  MISSING_METRICS — exit 0 but don't write metrics.json

Used for:
  - Demo 1 (tool failure): FAIL_ONCE or TIMEOUT
  - Demo 3 (goal drift):  WRONG_RESULT
  - Evaluation suite:     controlled failure injection per goal ID
"""
from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .runner import ExperimentConfig, ExecutionResult, ExperimentRunner


class FaultMode(str, Enum):
    NORMAL = "normal"
    """No injection — behaves like a real runner."""

    FAIL_ONCE = "fail_once"
    """Crash on first attempt for each experiment_id. Succeeds on retry."""

    TIMEOUT = "timeout"
    """Return a timed-out result (timed_out=True, exit_code=-1)."""

    CORRUPT_METRIC = "corrupt_metric"
    """
    Exit 0 and write metrics.json, but with accuracy inflated by a fixed
    offset that the independent evaluator will catch as inconsistent.
    """

    WRONG_RESULT = "wrong_result"
    """
    Return a result where accuracy is great but a constraint metric
    (e.g., latency) is violated — triggers goal drift detection.
    """

    MISSING_METRICS = "missing_metrics"
    """Exit 0 but don't write metrics.json — tests artifact collection robustness."""


class FaultInjectingRunner:
    """
    Wraps ExperimentRunner with per-experiment fault injection.

    Usage:
        runner = FaultInjectingRunner(
            base_runner=ExperimentRunner(base_dir=Path("runs/run_001/experiments")),
            fault_schedule={
                "exp_003": FaultMode.TIMEOUT,
                "exp_005": FaultMode.CORRUPT_METRIC,
            }
        )
        result = runner.run(config)  # exp_003 will time out
    """

    def __init__(
        self,
        base_runner: ExperimentRunner,
        fault_schedule: dict[str, FaultMode] | None = None,
        default_mode: FaultMode = FaultMode.NORMAL,
    ) -> None:
        self.base_runner = base_runner
        self.fault_schedule: dict[str, FaultMode] = fault_schedule or {}
        self.default_mode = default_mode
        # Track which experiments have already failed once (for FAIL_ONCE)
        self._failed_once: set[str] = set()
        self._call_index = 0

    def allocate_experiment_id(self, requested: str | None = None) -> str:
        return self.base_runner.allocate_experiment_id(requested)

    def run(self, config: ExperimentConfig) -> ExecutionResult:
        """Run the experiment, applying the configured fault mode."""
        self._call_index += 1
        mode = self._resolve_mode(config)
        if mode == FaultMode.FAIL_ONCE:
            return self._inject_fail_once(config)
        if mode == FaultMode.TIMEOUT:
            return self._inject_timeout(config)
        if mode == FaultMode.CORRUPT_METRIC:
            return self._inject_corrupt_metric(config)
        if mode == FaultMode.WRONG_RESULT:
            return self._inject_wrong_result(config)
        if mode == FaultMode.MISSING_METRICS:
            return self._inject_missing_metrics(config)
        return self.base_runner.run(config)

    def _resolve_mode(self, config: ExperimentConfig) -> FaultMode:
        first_aliases = {
            "1", "exp_01", "exp_1", "baseline_experiment_1",
            "baseline_experiment_001", "experiment_001",
        }
        schedule = {str(k): v for k, v in self.fault_schedule.items()}
        if config.experiment_id in self.fault_schedule:
            return self.fault_schedule[config.experiment_id]
        if str(self._call_index) in schedule:
            return schedule[str(self._call_index)]
        if self._call_index == 1:
            for alias in first_aliases:
                if alias in schedule:
                    return schedule[alias]
        return self.default_mode

    def collect_metrics(self, result: ExecutionResult) -> dict[str, Any]:
        return self.base_runner.collect_metrics(result)

    def experiment_dir(self, experiment_id: str) -> Path:
        return self.base_runner.experiment_dir(experiment_id)

    # ── Fault injection implementations ─────────────────────────────────────

    def _inject_fail_once(self, config: ExperimentConfig) -> ExecutionResult:
        """Crash first time, succeed second time."""
        exp_id = config.experiment_id
        if exp_id in self._failed_once:
            # Second attempt — run for real
            return self.base_runner.run(config)
        else:
            self._failed_once.add(exp_id)
            output_dir = self.base_runner.experiment_dir(exp_id)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "run.log").write_text(
                "=== STDERR ===\nRuntimeError: CUDA out of memory. "
                "Tried to allocate 256.00 MiB.\n"
            )
            return ExecutionResult(
                experiment_id=exp_id,
                exit_code=1,
                stdout="",
                stderr="RuntimeError: CUDA out of memory. Tried to allocate 256.00 MiB.",
                runtime_seconds=2.1,
                output_dir=output_dir,
                timed_out=False,
            )

    def _inject_timeout(self, config: ExperimentConfig) -> ExecutionResult:
        """Simulate process timeout."""
        output_dir = self.base_runner.experiment_dir(config.experiment_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run.log").write_text(
            "=== STDERR ===\nProcess timed out after "
            f"{config.timeout_seconds}s\n"
        )
        return ExecutionResult(
            experiment_id=config.experiment_id,
            exit_code=-1,
            stdout="Epoch 1/200 — acc: 0.31\nEpoch 2/200 — acc: 0.38\n",
            stderr=f"Process timed out after {config.timeout_seconds}s",
            runtime_seconds=float(config.timeout_seconds),
            output_dir=output_dir,
            timed_out=True,
        )

    def _inject_corrupt_metric(self, config: ExperimentConfig) -> ExecutionResult:
        """
        Run normally but write inflated accuracy to metrics.json.
        The training log will show the real (lower) value.
        This simulates the evaluator catching an inconsistency.
        """
        result = self.base_runner.run(config)
        if result.success:
            metrics_path = result.output_dir / "metrics.json"
            if metrics_path.exists():
                try:
                    metrics = json.loads(metrics_path.read_text())
                    real_acc = metrics.get("accuracy", 0.85)
                    # Inflate the metric — the log will show the real value
                    metrics["accuracy"] = min(real_acc + 0.08, 0.99)
                    metrics["_injected_fault"] = "corrupt_metric"
                    metrics_path.write_text(json.dumps(metrics, indent=2))
                except Exception:
                    pass
        return result

    def _inject_wrong_result(self, config: ExperimentConfig) -> ExecutionResult:
        """
        Return great accuracy but violate a constraint metric.
        Writes metrics.json with high accuracy + latency > threshold.
        Triggers goal_drift detection.
        """
        result = self.base_runner.run(config)
        output_dir = self.base_runner.experiment_dir(config.experiment_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "accuracy": 0.968,       # Great!
            "f1": 0.964,
            "loss": 0.11,
            "latency_ms": 143.7,     # Violates latency_under_100ms constraint
            "runtime_seconds": 18.4,
            "_injected_fault": "wrong_result",
        }
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        return ExecutionResult(
            experiment_id=config.experiment_id,
            exit_code=0,
            stdout="Training complete. accuracy=0.968 latency=143.7ms",
            stderr="",
            runtime_seconds=18.4,
            output_dir=output_dir,
            timed_out=False,
        )

    def _inject_missing_metrics(self, config: ExperimentConfig) -> ExecutionResult:
        """Exit 0 but don't write metrics.json."""
        output_dir = self.base_runner.experiment_dir(config.experiment_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "run.log").write_text("=== STDOUT ===\nTraining complete.\n")
        # Explicitly do NOT write metrics.json
        return ExecutionResult(
            experiment_id=config.experiment_id,
            exit_code=0,
            stdout="Training complete.",
            stderr="",
            runtime_seconds=8.2,
            output_dir=output_dir,
            timed_out=False,
        )
