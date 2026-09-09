"""
Experiment Runner — subprocess-based experiment execution.

Runs experiment scripts as isolated child processes with:
  - Configurable timeout
  - stdout/stderr capture to files
  - Metric collection from output JSON
  - Runtime measurement
  - Exit code checking

This is the bridge between the agent's typed tool calls and
the actual ML training scripts in environments/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    """Everything needed to run one experiment."""

    experiment_id: str
    script_path: Path
    output_dir: Path
    parameters: dict[str, Any] = field(default_factory=dict)
    timeout_seconds: int = 300
    seed: int = 42
    python_executable: str = sys.executable

    def to_args(self) -> list[str]:
        """Convert parameters dict to CLI args for the training script."""
        from ..agent.planning_context import (
            VALID_EXPERIMENT_PARAMS,
            alias_key,
            canonicalize_parameters,
            cli_flag_for,
        )

        args = [
            self.python_executable,
            str(self.script_path),
            "--output-dir", str(self.output_dir),
            "--seed", str(self.seed),
        ]
        params = canonicalize_parameters(self.parameters)
        for key, value in params.items():
            mapped = alias_key(key)
            if mapped not in VALID_EXPERIMENT_PARAMS:
                continue
            if mapped in {"seed"}:
                continue
            cli_key = cli_flag_for(mapped)
            if isinstance(value, bool):
                if value:
                    args.append(cli_key)
            else:
                args.extend([cli_key, str(value)])
        return args


@dataclass
class ExecutionResult:
    """Raw result of running an experiment process."""

    experiment_id: str
    exit_code: int
    stdout: str
    stderr: str
    runtime_seconds: float
    output_dir: Path
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def metrics_path(self) -> Path:
        return self.output_dir / "metrics.json"

    def log_path(self) -> Path:
        return self.output_dir / "run.log"


class ExperimentRunner:
    """
    Runs experiment scripts as isolated subprocesses.

    Usage:
        runner = ExperimentRunner(base_dir=Path("runs/run_001/experiments"))
        config = ExperimentConfig(
            experiment_id="exp_001",
            script_path=Path("environments/image_classification/train.py"),
            output_dir=runner.experiment_dir("exp_001"),
            parameters={"normalize": True, "lr": 0.01},
            seed=42,
        )
        result = runner.run(config)
        metrics = runner.collect_metrics(result)
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._seq = 0

    def experiment_dir(self, experiment_id: str) -> Path:
        d = self.base_dir / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def is_completed_experiment(path: Path) -> bool:
        return path.is_dir() and (
            (path / "metrics.json").exists() or (path / "config.json").exists()
        )

    def allocate_experiment_id(self, requested: str | None = None) -> str:
        """Allocate a unique ID. Planner-requested IDs are never trusted."""
        existing = {
            p.name
            for p in self.base_dir.iterdir()
            if p.is_dir() and self.is_completed_experiment(p)
        } if self.base_dir.exists() else set()
        if requested and requested not in existing:
            dest = self.base_dir / requested
            if not dest.exists() or not self.is_completed_experiment(dest):
                # Still ignore planner IDs so collisions cannot happen later.
                pass
        n = max(self._seq, 0)
        while True:
            n += 1
            eid = f"experiment_{n:03d}"
            dest = self.base_dir / eid
            if eid in existing or dest.exists():
                continue
            self._seq = n
            return eid

    def run(self, config: ExperimentConfig) -> ExecutionResult:
        """Execute the experiment script and capture all outputs."""
        output_dir = config.output_dir
        if self.is_completed_experiment(output_dir):
            raise FileExistsError(
                f"Refusing to overwrite completed experiment directory {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)

        from ..agent.planning_context import canonicalize_parameters, normalize_parameters
        canonical = normalize_parameters(canonicalize_parameters(config.parameters or {}))
        config_path = output_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "experiment_id": config.experiment_id,
                    "requested_experiment_id": getattr(config, "requested_id", None),
                    "parameters": canonical,
                    "seed": config.seed,
                    "timeout_seconds": config.timeout_seconds,
                    "latency_definition": (
                        "environment-defined computational-cost proxy, not measured serving latency"
                    ),
                },
                indent=2,
            )
        )

        args = config.to_args()
        start = time.monotonic()
        timed_out = False

        try:
            proc = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=config.timeout_seconds,
            )
            exit_code = proc.returncode
            stdout = proc.stdout
            stderr = proc.stderr
        except subprocess.TimeoutExpired as e:
            timed_out = True
            exit_code = -1
            stdout = e.stdout or ""
            stderr = str(e)
        except Exception as e:
            exit_code = -1
            stdout = ""
            stderr = str(e)

        runtime = time.monotonic() - start
        config_path.write_text(
            json.dumps(
                {
                    "experiment_id": config.experiment_id,
                    "parameters": canonical,
                    "seed": config.seed,
                    "timeout_seconds": config.timeout_seconds,
                    "latency_definition": (
                        "environment-defined computational-cost proxy, not measured serving latency"
                    ),
                },
                indent=2,
            )
        )

        (output_dir / "stdout.log").write_text(stdout or "", encoding="utf-8")
        (output_dir / "stderr.log").write_text(stderr or "", encoding="utf-8")
        log_path = output_dir / "run.log"
        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}\n")

        return ExecutionResult(
            experiment_id=config.experiment_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            runtime_seconds=round(runtime, 2),
            output_dir=output_dir,
            timed_out=timed_out,
        )

    def collect_metrics(self, result: ExecutionResult) -> dict[str, Any]:
        """
        Load metrics from the experiment's metrics.json output.
        Returns empty dict if the file doesn't exist or is malformed.
        """
        metrics_path = result.metrics_path()
        if not metrics_path.exists():
            return {}
        try:
            return json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
