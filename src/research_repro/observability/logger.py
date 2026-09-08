"""
JSONL event logger.

Appends one JSON line per event to `runs/<run_id>/run.jsonl`.
The log file IS the run — all analysis, visualization, and replay
derives from it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .events import AgentEvent, EventType


class EventLogger:
    """
    Appends structured AgentEvents to a JSONL file.

    Each call to .log() is synchronous and appends exactly one line.
    The step counter is authoritative — memory.step_count mirrors it.
    """

    def __init__(self, run_id: str, run_dir: Path) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.log_path = run_dir / "run.jsonl"
        self._step: int = 0
        run_dir.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: EventType, **payload: Any) -> AgentEvent:
        """
        Emit a structured event.

        Usage:
            logger.log(EventType.REASON, hypothesis="X", confidence=0.9)
        """
        self._step += 1
        event = AgentEvent(
            run_id=self.run_id,
            step=self._step,
            event_type=event_type,
            payload=payload,
        )
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        return event

    @property
    def step(self) -> int:
        return self._step
