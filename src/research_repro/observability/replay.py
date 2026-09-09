"""Replay persisted JSONL events without inventing run content."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_event_log(path: Path) -> list[dict]:
    if path.is_dir(): path = path / "run.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay_text(path: Path, failures_only: bool = False, recoveries_only: bool = False) -> str:
    events = load_event_log(path)
    if failures_only:
        allowed = {"failure_detected", "self_evaluation", "tool_result"}
        events = [e for e in events if e.get("event_type") in allowed]
    if recoveries_only:
        allowed = {"failure_detected", "recovery_started", "recovery_completed", "replan", "self_evaluation", "no_progress"}
        events = [e for e in events if e.get("event_type") in allowed]
    lines = ["ResearchRepro Run Replay", "=" * 32]
    for event in events:
        kind = event.get("event_type", "unknown").upper()
        payload = event.get("payload", {})
        lines += [f"\nStep {event.get('step', '?')} — {kind}", "-" * 32, json.dumps(payload, indent=2, default=str)]
    return "\n".join(lines)

