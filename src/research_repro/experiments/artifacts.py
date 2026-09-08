"""
Artifact Collector — gathers all outputs from a completed experiment.

After an experiment run, this collects:
  - metrics.json (primary)
  - run.log (full stdout/stderr)
  - config.json (hyperparameters used)
  - model files (if any)
  - Any other files in the output directory

All artifacts are catalogued with their paths and sizes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass
class Artifact:
    name: str
    path: Path
    size_bytes: int
    artifact_type: str  # "metrics" | "log" | "config" | "model" | "other"


@dataclass
class ExperimentArtifacts:
    experiment_id: str
    output_dir: Path
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def has_metrics(self) -> bool:
        return bool(self.metrics)

    def get_metric(self, name: str) -> float | None:
        return self.metrics.get(name)

    def summary(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "artifact_count": len(self.artifacts),
            "artifact_names": [a.name for a in self.artifacts],
            "metrics": self.metrics,
        }


class ArtifactCollector:
    """
    Scans an experiment output directory and catalogues all artifacts.

    Usage:
        collector = ArtifactCollector()
        artifacts = collector.collect(experiment_id="exp_001",
                                      output_dir=Path("runs/run_001/experiments/exp_001"))
        print(artifacts.metrics)  # {"accuracy": 0.912, "f1": 0.907, ...}
    """

    # File type classifications
    _TYPE_MAP = {
        "metrics.json": "metrics",
        "config.json": "config",
        "run.log": "log",
    }
    _MODEL_EXTENSIONS = {".pt", ".pth", ".pkl", ".joblib", ".h5", ".onnx"}

    def collect(self, experiment_id: str, output_dir: Path) -> ExperimentArtifacts:
        """Scan the output directory and return all artifacts with loaded metrics."""
        artifacts: list[Artifact] = []
        metrics: dict[str, Any] = {}

        if not output_dir.exists():
            return ExperimentArtifacts(
                experiment_id=experiment_id, output_dir=output_dir
            )

        for file_path in sorted(output_dir.iterdir()):
            if not file_path.is_file():
                continue

            artifact_type = self._classify(file_path)
            artifacts.append(
                Artifact(
                    name=file_path.name,
                    path=file_path,
                    size_bytes=file_path.stat().st_size,
                    artifact_type=artifact_type,
                )
            )

            # Load metrics from metrics.json
            if artifact_type == "metrics":
                try:
                    loaded = json.loads(file_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        metrics.update(loaded)
                except (json.JSONDecodeError, OSError):
                    pass

        return ExperimentArtifacts(
            experiment_id=experiment_id,
            output_dir=output_dir,
            artifacts=artifacts,
            metrics=metrics,
        )

    def _classify(self, path: Path) -> str:
        name = path.name
        if name in self._TYPE_MAP:
            return self._TYPE_MAP[name]
        if path.suffix in self._MODEL_EXTENSIONS:
            return "model"
        if "log" in name:
            return "log"
        return "other"
