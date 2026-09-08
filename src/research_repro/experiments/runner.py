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
        args = [
            self.python_executable,
            str(self.script_path),
            "--output-dir", str(self.output_dir),
            "--seed", str(self.seed),
        ]
        for key, value in self.parameters.items():
            cli_key = f"--{key.replace('_', '-')}"
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

    def experiment_dir(self, experiment_id: str) -> Path:
        d = self.base_dir / experiment_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run(self, config: ExperimentConfig) -> ExecutionResult:
        """Execute the experiment script and capture all outputs."""
        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save the config used (reproducibility)
        config_path = output_dir / "config.json"
        config_path.write_text(
            json.dumps(
                {
                    "experiment_id": config.experiment_id,
                    "parameters": config.parameters,
                    "seed": config.seed,
                    "timeout_seconds": config.timeout_seconds,
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

        # Write logs to disk
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
