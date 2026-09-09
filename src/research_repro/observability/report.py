"""Artifact-derived Markdown and HTML scientific run reports."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .replay import load_event_log


class ReportGenerator:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.memory = json.loads((run_dir / "memory.json").read_text(encoding="utf-8")) if (run_dir / "memory.json").exists() else {}
        self.events = load_event_log(run_dir / "run.jsonl") if (run_dir / "run.jsonl").exists() else []
        evidence_path = run_dir / "evidence.jsonl"
        self.evidence = [json.loads(x) for x in evidence_path.read_text(encoding="utf-8").splitlines() if x.strip()] if evidence_path.exists() else []

    def markdown(self) -> str:
        goal = self.memory.get("goal", {})
        reproduction = self.memory.get("reproduction", {})
        experiments = self.memory.get("experiments", [])
        failures = self.memory.get("failures", [])
        recoveries = self.memory.get("recoveries", [])
        lines = [
            f"# ResearchRepro Report: {self.memory.get('run_id', self.run_dir.name)}",
            "",
            "## Executive Summary", "",
            f"- **Goal:** {goal.get('objective', 'unknown')}",
            f"- **Verdict:** {reproduction.get('verdict', 'inconclusive').upper()}",
            f"- **Primary metric:** {goal.get('primary_metric', 'unknown')}",
            f"- **Experiments:** {len(experiments)}",
            f"- **Recoveries:** {len(recoveries)}",
            f"- **Unresolved issues:** {len(self.memory.get('unresolved_questions', []))}",
            "",
            "## Reproduction Assessment", "",
            "| Metric success | Methodology aligned | Independently verified | Verdict |",
            "|---|---|---|---|",
            f"| {reproduction.get('metric_success')} | {reproduction.get('methodology_aligned')} | {reproduction.get('independently_verified')} | {reproduction.get('verdict', 'inconclusive')} |",
            "",
            "## Experiment Lineage", "",
            "| Experiment | Parent | Hypothesis | Status | Result | Git commit |",
            "|---|---|---|---|---|---|",
        ]
        for e in experiments:
            result = e.get("observed_result") or {}
            lines.append(f"| {e.get('id')} | {e.get('parent_id') or 'BASELINE'} | {e.get('hypothesis','')} | {e.get('evaluation_status')} | {result} | {e.get('git_commit') or ''} |")
        lines += ["", "## Failure Audit", "", "| ID | Type | Experiment | Recovered | Description |", "|---|---|---|---|---|"]
        for f in failures:
            lines.append(f"| {f.get('id')} | {f.get('failure_type')} | {f.get('experiment_id') or ''} | {f.get('recovered')} | {f.get('description','')} |")
        lines += ["", "## Recovery Audit", "", "| Failure | Strategy | Diagnosis | Evidence | Plan change | Outcome |", "|---|---|---|---|---|---|"]
        for r in recoveries:
            lines.append(f"| {r.get('failure_id')} | {r.get('strategy')} | {r.get('diagnosis','')} | {r.get('evidence_used',[])} | {r.get('plan_change','')} | {r.get('success')} |")
        lines += ["", "## Evidence Ledger", ""]
        for evidence in self.evidence:
            lines.append(f"- {evidence.get('claim', evidence)} — source: {evidence.get('source_id', '')} {evidence.get('location', '')}")
        lines += ["", "## Honest Failure / Limitations", ""]
        unresolved = self.memory.get("unresolved_questions", [])
        lines += [f"- {x}" for x in unresolved] or ["- No unresolved questions were recorded."]
        return "\n".join(lines) + "\n"

    def html(self) -> str:
        body = html.escape(self.markdown())
        return f"<!doctype html><html><head><meta charset='utf-8'><title>ResearchRepro Report</title><style>body{{font-family:system-ui;max-width:1000px;margin:2rem auto;line-height:1.5}}pre{{white-space:pre-wrap}}</style></head><body><pre>{body}</pre></body></html>"

    def write(self, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(self.html() if output.suffix.lower() == ".html" else self.markdown(), encoding="utf-8")
        return output

