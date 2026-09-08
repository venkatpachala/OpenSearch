"""
Phase 8: Automated Chaos Harness for benchmark scorecards.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .chaos import FAULT_MATRIX, ChaosFault
from .runner import ExperimentRunner, ExperimentConfig
from ..recovery.strategies import STRATEGIES


class ChaosHarness:
    def __init__(self, runner: ExperimentRunner, script_path: Path, output_dir: Path):
        self.runner = runner
        self.script_path = script_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def benchmark(self) -> dict[str, Any]:
        cases = []
        passed_count = 0

        for fault in FAULT_MATRIX:
            strategy = STRATEGIES.get(fault.failure_type)
            has_matching_strategy = strategy is not None and strategy.name == fault.expected_strategy
            if has_matching_strategy:
                passed_count += 1

            cases.append({
                "id": fault.id,
                "failure_type": fault.failure_type.value,
                "expected_strategy": fault.expected_strategy,
                "actual_strategy": strategy.name if strategy else None,
                "passed": has_matching_strategy,
            })

        total = len(FAULT_MATRIX)
        score = passed_count / total if total else 0.0

        scorecard = {
            "total": total,
            "passed": passed_count,
            "score": score,
            "cases": cases,
        }

        (self.output_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2))
        return scorecard
