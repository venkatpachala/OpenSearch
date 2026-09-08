"""
CLI entry point — Typer-based commands.

Commands:
  run      Start a new research reproduction run
  show     View a run's event timeline
  list     List all runs
  status   Show current phase + budget for a run
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

app = typer.Typer(
    name="research-repro",
    help="Self-Correcting Scientific Experiment Agent",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    objective: str = typer.Argument(..., help="Research objective to pursue"),
    max_experiments: int = typer.Option(
        8, "--max-experiments", "-n", help="Max experiments allowed"
    ),
    target_metric: str = typer.Option(
        "accuracy", "--target-metric", help="Primary metric name"
    ),
    target_value: Optional[float] = typer.Option(
        None, "--target-value", help="Target metric value (e.g. 0.942)"
    ),
    success_threshold: float = typer.Option(
        1.0, "--threshold", help="Acceptable deviation from target (same units)"
    ),
    constraint: list[str] = typer.Option(
        [], "--constraint", "-c", help="Add constraint (e.g. 'latency_under_100ms')"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="LLM model name (default: qwen2.5-coder:7b)"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="LLM API base URL (default: http://localhost:11434/v1)"
    ),
    environment: str = typer.Option(
        "auto", "--environment", "-e", help="Environment to execute: 'auto', 'image_classification', or 'rag_retrieval'"
    ),
    stub_tools: bool = typer.Option(
        False, "--stub-tools", help="Use stub tools (no real ML, no GPU needed)"
    ),
    real_tools: bool = typer.Option(
        False, "--real-tools", help="Use real ML experiment tools (runs train.py or eval_rag.py via subprocess)"
    ),
    live_papers: bool = typer.Option(
        True, "--live-papers/--cached-papers",
        help="Use live arXiv/Semantic Scholar retrieval with local caching (real mode).",
    ),
    fault: list[str] = typer.Option(
        [], "--fault", "-f",
        help="Inject fault for experiment_id (format: 'exp_id:fault_mode'). "
             "Modes: fail_once, timeout, corrupt_metric, wrong_result, missing_metrics"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Parse goal and exit without running the agent"
    ),
) -> None:
    """Start a new research reproduction run."""
    from .config import Config
    from .memory.models import Constraint, GoalContract, ResourceBudget
    from .agent.loop import AgentLoop

    config = Config.from_env()
    config.use_stub_tools = stub_tools
    if model:
        config.openai_model = model
    if base_url:
        config.llm_base_url = base_url

    # Auto-route default metric if objective is about RAG / retrieval
    is_rag_objective = any(kw in objective.lower() for kw in ("rag", "rerank", "retrieval", "ndcg", "mrr", "passage"))
    if is_rag_objective and target_metric == "accuracy":
        target_metric = "ndcg@10"

    # Build goal contract
    constraints = [Constraint(name=c, description=c) for c in constraint]
    goal = GoalContract(
        objective=objective,
        primary_metric=target_metric,
        target_value=target_value,
        success_threshold=success_threshold,
        constraints=constraints,
        max_experiments=max_experiments,
        allowed_resources=ResourceBudget(
            max_experiments=max_experiments,
            max_recoveries_per_subtask=config.max_recoveries_per_subtask,
            max_total_recoveries=config.max_recoveries,
        ),
    )

    console.print(
        Panel(
            JSON(goal.model_dump_json(indent=2)),
            title="[bold cyan]Goal Contract[/bold cyan]",
            expand=False,
        )
    )

    if dry_run:
        console.print("[green]✓ Dry run complete — goal contract parsed successfully.[/green]")
        raise typer.Exit()

    # Validate tool mode
    if not stub_tools and not real_tools:
        console.print(
            "[yellow]Hint: Use --stub-tools (fast, no ML) or --real-tools (runs real subprocess).[/yellow]\n"
            "Defaulting to --stub-tools."
        )
        stub_tools = True

    # Check API key if using cloud OpenAI endpoint (not needed for localhost Ollama)
    is_local = "localhost" in config.llm_base_url or "127.0.0.1" in config.llm_base_url or "ollama" in config.llm_base_url
    if not is_local and not config.openai_api_key:
        console.print(
            "[red]Error: OPENAI_API_KEY is not set for remote endpoint.\n"
            "Set it in your .env file, or use local Ollama.[/red]"
        )
        raise typer.Exit(1)

    run_id = f"run_{int(time.time())}"
    console.print(f"\n[bold]Run ID:[/bold] [cyan]{run_id}[/cyan]")
    console.print(f"[bold]Model:[/bold]  [cyan]{config.openai_model}[/cyan] ({config.llm_base_url})")

    if stub_tools:
        from .tools.stub import build_stub_registry
        registry = build_stub_registry()
        console.print("[bold]Tools:[/bold]  stub (simulated fast loop)\n")
    else:
        # Parse fault schedule
        import re
        from .experiments.fault_injection import FaultMode
        fault_schedule: dict[str, FaultMode] = {}
        for f in fault:
            parts = f.split(":", 1)
            if len(parts) == 2:
                exp_id, mode_str = parts
                try:
                    fault_schedule[exp_id.strip()] = FaultMode(mode_str.strip())
                except ValueError:
                    console.print(f"[yellow]Unknown fault mode '{mode_str}', skipping.[/yellow]")

        # Determine environment
        project_root = Path(__file__).resolve().parent.parent.parent
        environments_dir = project_root / "environments"

        env_choice = environment.lower()
        if env_choice == "auto":
            env_choice = "rag_retrieval" if is_rag_objective else "image_classification"

        if env_choice == "rag_retrieval":
            script_path = environments_dir / "rag_retrieval" / "eval_rag.py"
            eval_script_path = environments_dir / "rag_retrieval" / "evaluate_rag.py"
            env_label = "RAG Retrieval & Reranking (eval_rag.py)"
        else:
            script_path = environments_dir / "image_classification" / "train.py"
            eval_script_path = environments_dir / "image_classification" / "evaluate.py"
            env_label = "Image Classification (train.py)"

        if not script_path.exists():
            console.print(f"[red]Environment script not found: {script_path}[/red]")
            raise typer.Exit(1)

        from .tools.execution import build_real_registry
        experiments_base = config.runs_dir / run_id / "experiments"
        registry = build_real_registry(
            experiments_base_dir=experiments_base,
            script_path=script_path,
            eval_script_path=eval_script_path,
            fault_schedule=fault_schedule or None,
            live_papers=live_papers,
        )
        mode_str = f"real subprocess execution: {env_label}"
        if fault_schedule:
            mode_str += f" + fault injection {fault_schedule}"
        console.print(f"[bold]Tools:[/bold]  {mode_str}\n")

    agent = AgentLoop(
        goal=goal,
        run_id=run_id,
        config=config,
        registry=registry,
    )

    with console.status("[bold green]Agent loop running...[/bold green]", spinner="dots"):
        report = agent.run()

    console.print(
        Panel(
            f"[bold]Completed:[/bold]        {report.completed}\n"
            f"[bold]Steps:[/bold]            {report.total_steps}\n"
            f"[bold]Experiments run:[/bold]  {report.experiments_run}\n"
            f"[bold]Self-corrections:[/bold] {report.self_corrections}\n"
            f"[bold]Recovery failures:[/bold]{report.recovery_failures}\n"
            f"[bold]Best accuracy:[/bold]    {report.best_accuracy}\n"
            f"[bold]Reported accuracy:[/bold]{report.reported_accuracy}\n"
            f"[bold]Gap:[/bold]              {report.gap}",
            title="[bold green]Final Report[/bold green]",
            expand=False,
        )
    )
    console.print(
        f"\nView timeline: [cyan]research-repro show {run_id}[/cyan]"
    )


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command()
def show(
    run_id: str = typer.Argument(..., help="Run ID to display"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", help="Runs directory"),
) -> None:
    """Display a run's event timeline."""
    from .observability.viewer import show_run

    try:
        show_run(run_id, runs_dir)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@app.command(name="list")
def list_runs(
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    """List all runs in the runs directory."""
    if not runs_dir.exists() or not any(runs_dir.iterdir()):
        console.print("[yellow]No runs found.[/yellow]")
        raise typer.Exit()

    runs = sorted(
        [r for r in runs_dir.iterdir() if r.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    console.print(f"\n[bold]{len(runs)} run(s) found:[/bold]\n")
    for run_path in runs:
        log = run_path / "run.jsonl"
        mem = run_path / "memory.json"
        phase = "?"
        if mem.exists():
            try:
                data = json.loads(mem.read_text())
                phase = data.get("current_phase", "?")
            except Exception:
                pass
        size = f"{log.stat().st_size:,} bytes" if log.exists() else "no log"
        console.print(f"  [cyan]{run_path.name}[/cyan]  phase={phase}  ({size})")
    console.print()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(
    run_id: str = typer.Argument(..., help="Run ID"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir"),
) -> None:
    """Show current phase and budget for a run."""
    memory_path = runs_dir / run_id / "memory.json"
    if not memory_path.exists():
        console.print(f"[red]Run '{run_id}' not found at {memory_path}[/red]")
        raise typer.Exit(1)

    data = json.loads(memory_path.read_text())
    budget = data.get("budget", {})
    goal = data.get("goal", {})
    resources = goal.get("allowed_resources", {})

    console.print(f"\n[bold]Run:[/bold]       {run_id}")
    console.print(f"[bold]Phase:[/bold]     {data.get('current_phase', '?')}")
    console.print(f"[bold]Steps:[/bold]     {data.get('step_count', 0)}")
    console.print(
        f"[bold]Experiments:[/bold] "
        f"{budget.get('experiments_consumed', 0)}/{goal.get('max_experiments', '?')}"
    )
    console.print(
        f"[bold]Recoveries:[/bold]  "
        f"{budget.get('recoveries_consumed', 0)}/{resources.get('max_total_recoveries', '?')}"
    )
    console.print()


@app.command(name="chaos-benchmark")
def chaos_benchmark(
    output_dir: Path = typer.Option(Path("runs/chaos_benchmark"), "--output-dir"),
    environment: str = typer.Option("image_classification", "--environment"),
) -> None:
    """Run the six controlled faults against a real environment and print a scorecard."""
    from .experiments import ChaosHarness
    from .experiments.runner import ExperimentRunner

    project_root = Path(__file__).resolve().parent.parent.parent
    env_dir = project_root / "environments" / environment
    script = env_dir / ("eval_rag.py" if environment == "rag_retrieval" else "train.py")
    if not script.exists():
        console.print(f"[red]Environment script not found: {script}[/red]")
        raise typer.Exit(1)
    runner = ExperimentRunner(output_dir / "experiments")
    scorecard = ChaosHarness(runner, script, output_dir).benchmark()
    console.print(JSON(json.dumps(scorecard)))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
