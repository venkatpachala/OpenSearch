"""
Real execution tools — use ExperimentRunner + actual training scripts.

These replace the stub tools when --stub-tools is NOT passed.
Same tool interface as stubs — the agent loop doesn't know the difference.

Phase 4: These tools actually run environments/image_classification/train.py
and collect real metrics from the subprocess output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import Tool, ToolRequest, ToolResponse, ToolRegistry
from ..experiments.runner import ExperimentConfig, ExperimentRunner
from ..experiments.artifacts import ArtifactCollector
from ..experiments.fault_injection import FaultInjectingRunner, FaultMode
from ..experiments.checkpoints import GitCheckpointer


# ---------------------------------------------------------------------------
# run_experiment (real)
# ---------------------------------------------------------------------------


class RunExperimentRequest(ToolRequest):
    experiment_id: str
    parameters: dict = {}
    timeout_seconds: int = 600
    seed: int = 42
    parent_experiment_id: str | None = None


class RealRunExperimentTool(Tool):
    name = "run_experiment"
    description = (
        "Execute a real training experiment with the given parameters. "
        "Runs the training script as a subprocess and collects metrics. "
        "Use experiment_id to uniquely identify this run. Set seed for reproducibility."
    )
    request_model = RunExperimentRequest

    def __init__(
        self,
        runner: ExperimentRunner | FaultInjectingRunner,
        script_path: Path,
        checkpointer: GitCheckpointer | None = None,
    ) -> None:
        self.runner = runner
        self.script_path = script_path
        self.checkpointer = checkpointer
        self.collector = ArtifactCollector()

    def execute(self, request: RunExperimentRequest) -> ToolResponse:
        from ..agent.planning_context import canonicalize_parameters, normalize_parameters

        requested = request.experiment_id
        allocator = getattr(self.runner, "allocate_experiment_id", None)
        experiment_id = allocator(requested) if callable(allocator) else requested
        output_dir = self.runner.experiment_dir(experiment_id)
        canonical = normalize_parameters(canonicalize_parameters(request.parameters or {}))
        config = ExperimentConfig(
            experiment_id=experiment_id,
            script_path=self.script_path,
            output_dir=output_dir,
            parameters=canonical,
            timeout_seconds=request.timeout_seconds,
            seed=request.seed,
        )

        result = self.runner.run(config)
        artifacts = self.collector.collect(experiment_id, output_dir)
        hypothesis_path = output_dir / "hypothesis.json"
        if not hypothesis_path.exists():
            hypothesis_path.write_text(
                json.dumps(
                    {
                        "hypothesis": getattr(request, "hypothesis", None),
                        "requested_experiment_id": requested,
                        "experiment_id": experiment_id,
                        "canonical_configuration": canonical,
                    },
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )

        if result.timed_out:
            return ToolResponse.fail(
                f"Experiment timed out after {request.timeout_seconds}s. "
                f"Partial stdout: {result.stdout[:200]}"
            )

        if not result.success:
            return ToolResponse.fail(
                f"Experiment failed (exit code {result.exit_code}). "
                f"stderr: {result.stderr[:300]}"
            )

        if not artifacts.has_metrics:
            return ToolResponse.fail(
                "Experiment completed but metrics.json was not written. "
                f"stdout: {result.stdout[:300]}"
            )

        commit_hash = None
        if self.checkpointer is not None:
            commit_hash = self.checkpointer.checkpoint(
                experiment_id=experiment_id,
                message=f"Experiment {experiment_id} completed",
            )

        return ToolResponse.ok(
            experiment_id=experiment_id,
            requested_experiment_id=requested,
            accuracy=artifacts.get_metric("accuracy"),
            f1=artifacts.get_metric("f1"),
            loss=artifacts.get_metric("loss"),
            runtime_seconds=artifacts.get_metric("runtime_seconds") or result.runtime_seconds,
            latency_ms=artifacts.get_metric("latency_ms"),
            iterations=artifacts.get_metric("iterations"),
            artifacts=[str(a.path) for a in artifacts.artifacts],
            stdout_summary=result.stdout[-400:],
            metrics=artifacts.metrics,
            git_commit=commit_hash,
        )


# ---------------------------------------------------------------------------
# collect_metrics (real)
# ---------------------------------------------------------------------------


class CollectMetricsRequest(ToolRequest):
    experiment_id: str
    metric_names: list[str] = ["accuracy", "f1", "loss"]


class RealCollectMetricsTool(Tool):
    name = "collect_metrics"
    description = (
        "Collect metrics from a completed experiment's output directory. "
        "Use after run_experiment to retrieve specific metric values."
    )
    request_model = CollectMetricsRequest

    def __init__(self, experiments_base_dir: Path) -> None:
        self.experiments_base_dir = experiments_base_dir
        self.collector = ArtifactCollector()

    def execute(self, request: CollectMetricsRequest) -> ToolResponse:
        output_dir = self.experiments_base_dir / request.experiment_id
        artifacts = self.collector.collect(request.experiment_id, output_dir)

        if not artifacts.has_metrics:
            return ToolResponse.fail(
                f"No metrics found for experiment '{request.experiment_id}'. "
                "Run run_experiment first."
            )

        selected = {
            k: v
            for k, v in artifacts.metrics.items()
            if k in request.metric_names or not request.metric_names
        }
        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            metrics=selected,
            all_metrics=artifacts.metrics,
        )


# ---------------------------------------------------------------------------
# compare_results (same interface as stub, no real ML dependency)
# ---------------------------------------------------------------------------


class CompareResultsRequest(ToolRequest):
    experiment_id: str
    reported_value: float | None = None
    observed_value: float | None = None
    reported_accuracy: float | None = None
    observed_accuracy: float | None = None
    threshold: float = 0.005


class CompareResultsTool(Tool):
    name = "compare_results"
    description = (
        "Compare observed experiment results against paper's reported results. "
        "Supports any metric (accuracy, ndcg@10, relative_improvement). "
        "Returns status (REPRODUCED or DISCREPANCY) and the absolute gap."
    )
    request_model = CompareResultsRequest

    def execute(self, request: CompareResultsRequest) -> ToolResponse:
        rep = request.reported_value if request.reported_value is not None else (request.reported_accuracy or 0.0)
        obs = request.observed_value if request.observed_value is not None else (request.observed_accuracy or 0.0)

        diff = abs(rep - obs)
        within = diff <= request.threshold
        pct = diff / max(rep, 1e-9) * 100
        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            reported=round(rep, 4),
            observed=round(obs, 4),
            absolute_difference=round(diff, 4),
            relative_difference_pct=round(pct, 2),
            within_threshold=within,
            status="REPRODUCED" if within else "DISCREPANCY",
        )


# ---------------------------------------------------------------------------
# run_independent_evaluation (Phase 4 addition)
# ---------------------------------------------------------------------------


class RunIndependentEvalRequest(ToolRequest):
    experiment_id: str
    timeout_seconds: int = 120
    seed: int = 1042  # Different from training seed


class RunIndependentEvalTool(Tool):
    name = "run_independent_evaluation"
    description = (
        "Run an independent evaluation of a completed experiment using a separate "
        "evaluation script. Use this to verify training metrics and detect "
        "result inconsistencies (Failure Type 3)."
    )
    request_model = RunIndependentEvalRequest

    def __init__(
        self,
        runner: ExperimentRunner,
        eval_script_path: Path,
    ) -> None:
        self.runner = runner
        self.eval_script_path = eval_script_path
        self.collector = ArtifactCollector()

    def execute(self, request: RunIndependentEvalRequest) -> ToolResponse:
        model_dir = self.runner.experiment_dir(request.experiment_id)
        eval_output_dir = model_dir / "eval"
        eval_output_dir.mkdir(parents=True, exist_ok=True)

        import sys
        config = ExperimentConfig(
            experiment_id=f"{request.experiment_id}_eval",
            script_path=self.eval_script_path,
            output_dir=eval_output_dir,
            parameters={
                "model-dir": str(model_dir),
                "output-dir": str(eval_output_dir),
            },
            timeout_seconds=request.timeout_seconds,
            seed=request.seed,
        )

        # Build custom args (evaluate.py has different arg structure)
        import subprocess
        result_proc = subprocess.run(
            [
                sys.executable,
                str(self.eval_script_path),
                "--model-dir", str(model_dir),
                "--output-dir", str(eval_output_dir),
                "--seed", str(request.seed),
            ],
            capture_output=True,
            text=True,
            timeout=request.timeout_seconds,
        )

        if result_proc.returncode != 0:
            return ToolResponse.fail(
                f"Independent evaluation failed: {result_proc.stderr[:300]}"
            )

        eval_metrics_path = eval_output_dir / "eval_metrics.json"
        if not eval_metrics_path.exists():
            return ToolResponse.fail("Evaluation script did not produce eval_metrics.json")

        import json
        eval_metrics = json.loads(eval_metrics_path.read_text())
        return ToolResponse.ok(
            experiment_id=request.experiment_id,
            eval_accuracy=eval_metrics.get("eval_accuracy"),
            training_accuracy=eval_metrics.get("training_accuracy"),
            discrepancy=eval_metrics.get("discrepancy"),
            consistent=eval_metrics.get("consistent"),
            independently_verified=eval_metrics.get("independently_verified"),
            limitation=eval_metrics.get("limitation"),
            eval_metrics=eval_metrics,
        )


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------


def build_real_registry(
    experiments_base_dir: Path,
    script_path: Path,
    eval_script_path: Path,
    fault_schedule: dict[str, FaultMode] | None = None,
    live_papers: bool = True,
    default_fault_mode: FaultMode | None = None,
) -> ToolRegistry:
    """
    Build and return a ToolRegistry with real execution tools.

    Args:
        experiments_base_dir: Where to store experiment outputs.
        script_path:          Path to environments/image_classification/train.py
        eval_script_path:     Path to environments/image_classification/evaluate.py
        fault_schedule:       Optional dict of experiment_id → FaultMode for injection.
    """
    from .stub import InspectMethodologyTool

    base_runner = ExperimentRunner(base_dir=experiments_base_dir)

    if fault_schedule:
        runner = FaultInjectingRunner(
            base_runner=base_runner,
            fault_schedule=fault_schedule,
            default_mode=default_fault_mode or FaultMode.NORMAL,
        )
    else:
        runner = base_runner

    checkpointer = GitCheckpointer(workspace_dir=experiments_base_dir)
    checkpointer.initialize()

    registry = ToolRegistry()

    if live_papers:
        from .research import LiveSearchLiteratureTool, LiveRetrievePaperTool, LiveExtractMethodologyTool
        research_cache = experiments_base_dir.parent.parent / "cache"
        registry.register(LiveSearchLiteratureTool(research_cache))
        registry.register(LiveRetrievePaperTool(research_cache))
        registry.register(LiveExtractMethodologyTool())
    else:
        from .stub import SearchLiteratureTool, RetrievePaperTool, ExtractMethodologyTool
        registry.register(SearchLiteratureTool())
        registry.register(RetrievePaperTool())
        registry.register(ExtractMethodologyTool())
    registry.register(InspectMethodologyTool())

    # Real execution tools
    registry.register(RealRunExperimentTool(runner=runner, script_path=script_path, checkpointer=checkpointer))
    registry.register(RealCollectMetricsTool(experiments_base_dir=experiments_base_dir))
    registry.register(CompareResultsTool())
    registry.register(RunIndependentEvalTool(
        runner=base_runner,
        eval_script_path=eval_script_path,
    ))

    return registry
