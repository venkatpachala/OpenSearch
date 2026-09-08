"""
Evidence Ledger — traceable claims with typed recording methods.

Every important claim the agent makes or discovers is recorded here
with a source citation. The final report can therefore say:

    Claim: "Paper uses cosine LR schedule"
    Source: paper_001, page=7, section='Training Details'

    Claim: "Experiment 06 achieved 93.9%"
    Source: experiment_06/metrics.json, git_commit=abc123

This makes the system honest — it cannot report facts without evidence.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import Evidence, ResearchMemory


class EvidenceLedger:
    """
    Typed evidence recording with automatic JSONL persistence.

    All evidence is stored in ResearchMemory.evidence (in-memory) AND
    appended to runs/<run_id>/evidence.jsonl (on disk) for durability.
    """

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.ledger_path = run_dir / "evidence.jsonl"
        run_dir.mkdir(parents=True, exist_ok=True)

    # ── Typed recording methods ──────────────────────────────────────────────

    def record_paper_claim(
        self,
        memory: ResearchMemory,
        claim: str,
        paper_id: str,
        page: int | None = None,
        section: str = "",
    ) -> Evidence:
        """Record a claim extracted from a paper (e.g., reported metrics, methodology)."""
        location_parts = []
        if page is not None:
            location_parts.append(f"page={page}")
        if section:
            location_parts.append(f"section='{section}'")
        location = ", ".join(location_parts)

        evidence = Evidence(
            claim=claim,
            source_type="paper",
            source_id=paper_id,
            location=location,
        )
        self._record(memory, evidence)
        return evidence

    def record_experiment_result(
        self,
        memory: ResearchMemory,
        claim: str,
        experiment_id: str,
        metric_file: str = "metrics.json",
        git_commit: str = "",
    ) -> Evidence:
        """Record a claim backed by observed experiment results."""
        location_parts = [metric_file]
        if git_commit:
            location_parts.append(f"git_commit={git_commit[:8]}")
        location = ", ".join(location_parts)

        evidence = Evidence(
            claim=claim,
            source_type="experiment",
            source_id=experiment_id,
            location=location,
        )
        self._record(memory, evidence)
        return evidence

    def record_observation(
        self,
        memory: ResearchMemory,
        claim: str,
        step_index: int,
        tool_name: str,
    ) -> Evidence:
        """Record a claim based on direct agent observation during a tool call."""
        evidence = Evidence(
            claim=claim,
            source_type="observation",
            source_id=f"step_{step_index}",
            location=f"tool={tool_name}",
        )
        self._record(memory, evidence)
        return evidence

    def record_discrepancy(
        self,
        memory: ResearchMemory,
        metric: str,
        expected: float,
        observed: float,
        experiment_id: str,
    ) -> Evidence:
        """Record a detected discrepancy as evidence."""
        gap = abs(expected - observed)
        pct = gap / max(expected, 1e-9) * 100
        claim = (
            f"Discrepancy detected for '{metric}': "
            f"expected={expected:.4f}, observed={observed:.4f}, "
            f"gap={gap:.4f} ({pct:.1f}%)"
        )
        evidence = Evidence(
            claim=claim,
            source_type="observation",
            source_id=experiment_id,
            location=f"metric={metric}",
        )
        self._record(memory, evidence)
        return evidence

    # ── Query helpers ────────────────────────────────────────────────────────

    def get_evidence_for(
        self, memory: ResearchMemory, source_type: str | None = None
    ) -> list[Evidence]:
        """Return evidence filtered by source type."""
        items = memory.evidence
        if source_type:
            items = [e for e in items if e.source_type == source_type]
        return items

    def to_report_section(self, memory: ResearchMemory) -> str:
        """Render the evidence ledger as a formatted report section."""
        if not memory.evidence:
            return "No evidence recorded."

        lines = ["## Evidence Ledger\n"]
        by_type: dict[str, list[Evidence]] = {}
        for ev in memory.evidence:
            by_type.setdefault(ev.source_type, []).append(ev)

        for source_type, items in by_type.items():
            lines.append(f"### {source_type.title()} Evidence\n")
            for ev in items:
                lines.append(f"- **{ev.claim}**")
                lines.append(f"  Source: {ev.source_id} | {ev.location}")
                lines.append(f"  Recorded: {ev.recorded_at.strftime('%H:%M:%S')}\n")

        return "\n".join(lines)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _record(self, memory: ResearchMemory, evidence: Evidence) -> None:
        """Add evidence to memory and append to JSONL."""
        memory.evidence.append(evidence)
        with self.ledger_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": evidence.recorded_at.isoformat(),
                        "claim": evidence.claim,
                        "source_type": evidence.source_type,
                        "source_id": evidence.source_id,
                        "location": evidence.location,
                    }
                )
                + "\n"
            )
