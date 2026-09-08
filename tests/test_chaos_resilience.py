"""
Tests for Phase 8: Chaos resilience & fault injection matrix.
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from research_repro.experiments.chaos import FAULT_MATRIX, ChaosFault
from research_repro.experiments.fault_injection import FaultInjectingRunner, FaultMode
from research_repro.experiments.runner import ExperimentRunner, ExperimentConfig
from research_repro.evaluation.failure_taxonomy import FailureTaxonomyClassifier
from research_repro.memory.models import FailureType
from research_repro.recovery.strategies import STRATEGIES


class TestChaosMatrix:
    def test_all_scenarios_have_matching_strategy(self):
        for scenario in FAULT_MATRIX:
            strategy = STRATEGIES.get(scenario.failure_type)
            assert strategy is not None
            assert strategy.name == scenario.expected_strategy

    def test_chaos_tool_crash_detection(self, tmp_path):
        base = ExperimentRunner(base_dir=tmp_path)
        fault_runner = FaultInjectingRunner(
            base_runner=base,
            fault_schedule={"exp_crash": FaultMode.FAIL_ONCE},
        )
        dummy_script = tmp_path / "script.py"
        dummy_script.write_text("print('ok')")

        cfg = ExperimentConfig(
            experiment_id="exp_crash",
            script_path=dummy_script,
            output_dir=tmp_path / "exp_crash",
        )
        result = fault_runner.run(cfg)
        assert not result.success

        # Classify
        classifier = FailureTaxonomyClassifier()
        taxonomy = classifier.classify("run_experiment", {"success": False, "error": result.stderr})
        assert taxonomy.is_failure
        assert taxonomy.failure_type == FailureType.TOOL_CRASH

    def test_chaos_goal_drift_detection(self):
        classifier = FailureTaxonomyClassifier()
        goal_facts = {
            "metric_achieved": True,
            "any_constraint_violated": True,
            "constraint_details": [{"name": "latency_under_100ms", "satisfied": False}],
        }
        taxonomy = classifier.classify("run_experiment", {}, goal_check_facts=goal_facts)
        assert taxonomy.is_failure
        assert taxonomy.failure_type == FailureType.GOAL_DRIFT

    def test_chaos_result_inconsistency_detection(self):
        classifier = FailureTaxonomyClassifier()
        tool_response = {
            "success": True,
            "data": {
                "eval_accuracy": 0.85,
                "training_accuracy": 0.95,
                "discrepancy": 0.10,
                "consistent": False,
            },
        }
        taxonomy = classifier.classify("run_independent_evaluation", tool_response)
        assert taxonomy.is_failure
        assert taxonomy.failure_type == FailureType.RESULT_INCONSISTENCY
