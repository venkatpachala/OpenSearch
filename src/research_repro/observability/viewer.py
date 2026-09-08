"""
CLI run viewer.

Reads a run's JSONL log and renders a Rich event timeline table
plus a summary of key statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# Icons for each event type
_ICONS: dict[str, str] = {
    "goal_set": "🎯",
    "plan_created": "📋",
    "phase_transition": "➡️",
    "reason": "🧠",
    "action": "⚡",
    "tool_call": "🔧",
    "tool_result": "📊",
    "observation": "👁",
    "self_evaluation": "⚖️",
    "tool_schema_error": "🚫",
    "failure_detected": "❌",
    "recovery_started": "🔄",
    "recovery_completed": "✅",
    "replan": "🔀",
    "experiment_started": "🧪",
    "experiment_completed": "🏁",
    "evidence_recorded": "📌",
    "discrepancy_detected": "⚠️",
    "budget_warning": "⚠️",
    "budget_exhausted": "🛑",
    "final_report": "📝",
}


def load_events(run_id: str, runs_dir: Path) -> list[dict]:
    """Read all events from a run's JSONL log."""
    log_path = runs_dir / run_id / "run.jsonl"
    if not log_path.exists():
        raise FileNotFoundError(
            f"No log found for run '{run_id}' at {log_path}. "
            "Run the agent first with: research-repro run \"...\""
        )
    events = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def show_run(run_id: str, runs_dir: Path) -> None:
    """Render a Rich timeline table for the given run."""
    events = load_events(run_id, runs_dir)

    console.print()
    console.print(Panel(f"[bold cyan]{run_id}[/bold cyan]", title="Run", expand=False))
    console.print()

    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("#", style="dim", width=5, justify="right")
    table.add_column("Event", width=26)
    table.add_column("Details", width=62)
    table.add_column("Time", style="dim", width=10)

    for ev in events:
        event_type = ev.get("event_type", "")
        icon = _ICONS.get(event_type, "•")
        payload = ev.get("payload", {})
        detail = _format_detail(event_type, payload)
        ts = ev.get("timestamp", "")[:19].replace("T", " ")

        # Color critical events
        row_style = ""
        if event_type in ("failure_detected", "tool_schema_error"):
            row_style = "red"
        elif event_type in ("recovery_started", "replan"):
            row_style = "yellow"
        elif event_type in ("recovery_completed", "final_report"):
            row_style = "green"

        table.add_row(
            str(ev.get("step", "")),
            f"{icon} {event_type}",
            detail,
            ts,
            style=row_style,
        )

    console.print(table)
    _print_summary(events)


def _format_detail(event_type: str, payload: dict) -> str:
    """Produce a compact one-line detail string for the table."""
    if event_type == "reason":
        return _truncate(payload.get("hypothesis", payload.get("intended_action", "")), 60)
    elif event_type == "tool_call":
        args = _short_dict(payload.get("args", {}))
        return f"{payload.get('tool_name', '')}({args})"
    elif event_type == "tool_result":
        success = "✓" if payload.get("success") else "✗"
        err = payload.get("error", "")
        return f"{success} {err[:50] if err else ''}"
    elif event_type == "self_evaluation":
        return (
            f"progress={payload.get('goal_progress', '?')}  "
            f"decision={payload.get('decision', '?')}  "
            f"conf={payload.get('confidence', '?')}"
        )
    elif event_type == "failure_detected":
        t = payload.get("failure_type") or payload.get("type", "?")
        desc = payload.get("description", "")[:40]
        return f"[{t}] {desc}"
    elif event_type == "phase_transition":
        return f"{payload.get('from_phase', '?')} → {payload.get('to_phase', '?')}"
    elif event_type == "experiment_completed":
        result = payload.get("result", {})
        return f"accuracy={result.get('accuracy', '?')}"
    elif event_type == "plan_created":
        steps = payload.get("steps", [])
        return f"{len(steps)} steps"
    elif event_type == "goal_set":
        goal = payload.get("goal", {})
        return _truncate(goal.get("objective", ""), 60)
    elif event_type == "tool_schema_error":
        n = len(payload.get("field_errors", []))
        return f"{n} field error(s) in {payload.get('tool_name', '?')}"
    return _truncate(str(payload), 60)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _short_dict(d: dict) -> str:
    s = str(d)
    return _truncate(s, 40)


def _print_summary(events: list[dict]) -> None:
    """Print aggregate statistics below the table."""
    corrections = sum(1 for e in events if e.get("event_type") == "recovery_started")
    failures = sum(1 for e in events if e.get("event_type") == "failure_detected")
    experiments = sum(1 for e in events if e.get("event_type") == "experiment_started")
    schema_errors = sum(1 for e in events if e.get("event_type") == "tool_schema_error")

    console.print()
    console.print(f"  [bold]Total events:[/bold]     {len(events)}")
    console.print(f"  [bold]Experiments run:[/bold]  {experiments}")
    console.print(f"  [bold]Self-corrections:[/bold] {corrections}")
    console.print(f"  [bold]Failures:[/bold]         {failures}")
    console.print(f"  [bold]Schema errors:[/bold]    {schema_errors}")
    console.print()
