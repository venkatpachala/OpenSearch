"""
Persistence layer for ResearchMemory.

Saves/loads the full agent state as JSON to the run directory.
Intentionally simple — no database. One file per run.
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import ResearchMemory


class MemoryStore:
    """Reads and writes ResearchMemory to/from a JSON file."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.path = run_dir / "memory.json"
        run_dir.mkdir(parents=True, exist_ok=True)

    def save(self, memory: ResearchMemory) -> None:
        """Serialize and write the full memory state."""
        self.path.write_text(
            memory.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def load(self) -> ResearchMemory:
        """Load memory from disk. Raises FileNotFoundError if not found."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"No memory snapshot found at {self.path}. "
                "Has this run been initialized?"
            )
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        return ResearchMemory(**raw)

    def exists(self) -> bool:
        return self.path.exists()
