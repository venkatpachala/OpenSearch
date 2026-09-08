"""
Git Checkpointer — experiment versioning via git.

Before each experiment modification, we commit the current state.
This enables:
  - Restoring to any known-good checkpoint
  - Comparing changes between experiments (git diff)
  - Auditing exactly what changed between EXP-01 and EXP-02

The experiment directory is treated as a mini git repo.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


class GitCheckpointer:
    """
    Manages git checkpoints for the experiment workspace.

    Each checkpoint is a git commit tagged with the experiment ID.
    The agent can restore to any previous checkpoint if a recovery
    strategy requires it (e.g., after a regression).
    """

    def __init__(self, workspace_dir: Path) -> None:
        self.workspace_dir = workspace_dir

    def initialize(self) -> bool:
        """Initialize a git repo in the workspace if not already present."""
        git_dir = self.workspace_dir / ".git"
        if git_dir.exists():
            return True
        result = self._git("init")
        if result.returncode != 0:
            return False
        # Set identity for commits (needed in some CI environments)
        self._git("config", "user.email", "agent@research-repro.local")
        self._git("config", "user.name", "ResearchRepro Agent")
        # Create initial commit if there are files
        self._git("add", ".")
        self._git("commit", "--allow-empty", "-m", "Initial checkpoint")
        return True

    def checkpoint(self, experiment_id: str, message: str = "") -> Optional[str]:
        """
        Create a git commit checkpoint for the given experiment.
        Returns the commit hash, or None if git is not available.
        """
        if not self._git_available():
            return None
        self._git("add", ".")
        commit_msg = f"checkpoint: {experiment_id}"
        if message:
            commit_msg += f" — {message}"
        self._git("commit", "--allow-empty", "-m", commit_msg)
        return self._get_head_hash()

    def restore(self, commit_hash: str) -> bool:
        """Restore the workspace to a specific git commit."""
        if not self._git_available():
            return False
        result = self._git("checkout", commit_hash, "--", ".")
        return result.returncode == 0

    def get_diff(self, from_hash: str, to_hash: str = "HEAD") -> str:
        """Get a diff between two checkpoints."""
        if not self._git_available():
            return ""
        result = self._git("diff", from_hash, to_hash)
        return result.stdout if result.returncode == 0 else ""

    def log(self, n: int = 10) -> list[dict]:
        """Return the last N checkpoint entries."""
        if not self._git_available():
            return []
        result = self._git(
            "log", f"-{n}", "--pretty=format:%H|%s|%ai", "--no-walk"
        )
        if result.returncode != 0:
            return []
        entries = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                entries.append({"hash": parts[0], "message": parts[1], "date": parts[2]})
        return entries

    # ── Internals ────────────────────────────────────────────────────────────

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", *args],
            cwd=self.workspace_dir,
            capture_output=True,
            text=True,
        )

    def _git_available(self) -> bool:
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_head_hash(self) -> Optional[str]:
        result = self._git("rev-parse", "HEAD")
        if result.returncode == 0:
            return result.stdout.strip()
        return None
