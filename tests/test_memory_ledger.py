"""
Tests for Phase 3: Evidence Ledger and enhanced memory integration.
"""
import tempfile
from pathlib import Path

import pytest

from research_repro.memory.ledger import EvidenceLedger
from research_repro.memory.models import (
    GoalContract, ResearchMemory, Evidence
)


def _make_memory() -> ResearchMemory:
    goal = GoalContract(
        objective="Reproduce MNIST classification",
        primary_metric="accuracy",
        target_value=0.97,
    )
    return ResearchMemory(goal=goal, run_id="test_ledger_run")


class TestEvidenceLedger:
    def test_record_paper_claim(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        ev = ledger.record_paper_claim(
            memory=mem,
            claim="Paper reports accuracy=97.0%",
            paper_id="paper_001",
            page=7,
            section="Results",
        )
        assert ev.source_type == "paper"
        assert ev.source_id == "paper_001"
        assert "page=7" in ev.location
        assert "Results" in ev.location
        assert len(mem.evidence) == 1

    def test_record_experiment_result(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        ev = ledger.record_experiment_result(
            memory=mem,
            claim="Experiment exp_001 achieved accuracy=0.932",
            experiment_id="exp_001",
            git_commit="abc123def456",
        )
        assert ev.source_type == "experiment"
        assert "abc123de" in ev.location  # truncated to 8 chars
        assert len(mem.evidence) == 1

    def test_record_observation(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        ev = ledger.record_observation(
            memory=mem,
            claim="Tool returned 5 papers",
            step_index=3,
            tool_name="search_literature",
        )
        assert ev.source_type == "observation"
        assert ev.source_id == "step_3"
        assert "search_literature" in ev.location

    def test_record_discrepancy(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        ev = ledger.record_discrepancy(
            memory=mem,
            metric="accuracy",
            expected=0.97,
            observed=0.92,
            experiment_id="exp_001",
        )
        assert "Discrepancy" in ev.claim
        assert "accuracy" in ev.claim
        assert "0.9700" in ev.claim
        assert "0.9200" in ev.claim

    def test_jsonl_persistence(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        ledger.record_paper_claim(mem, "Claim 1", "p1")
        ledger.record_paper_claim(mem, "Claim 2", "p2")

        ledger_path = tmp_path / "evidence.jsonl"
        assert ledger_path.exists()
        lines = ledger_path.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_get_evidence_for_source_type(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        ledger.record_paper_claim(mem, "Paper claim", "p1")
        ledger.record_observation(mem, "Observation", 1, "search_literature")
        ledger.record_experiment_result(mem, "Exp result", "exp_001")

        paper_evs = ledger.get_evidence_for(mem, source_type="paper")
        obs_evs = ledger.get_evidence_for(mem, source_type="observation")
        exp_evs = ledger.get_evidence_for(mem, source_type="experiment")

        assert len(paper_evs) == 1
        assert len(obs_evs) == 1
        assert len(exp_evs) == 1

    def test_to_report_section(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        ledger.record_paper_claim(mem, "Reported accuracy=0.97", "p1", page=7)

        report = ledger.to_report_section(mem)
        assert "Evidence Ledger" in report
        assert "Reported accuracy=0.97" in report

    def test_empty_report_section(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        report = ledger.to_report_section(mem)
        assert "No evidence" in report

    def test_multiple_recordings_accumulate(self, tmp_path):
        mem = _make_memory()
        ledger = EvidenceLedger(tmp_path)
        for i in range(5):
            ledger.record_paper_claim(mem, f"Claim {i}", f"paper_{i}")
        assert len(mem.evidence) == 5
