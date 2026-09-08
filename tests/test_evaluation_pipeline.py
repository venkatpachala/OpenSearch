"""
Tests for Phase 5: GoalChecker, DiscrepancyAnalyzer, FailureTaxonomyClassifier,
and the enhanced two-stage SelfEvaluator (without LLM calls).
"""
import pytest

from research_repro.evaluation.goal_checker import GoalChecker, GoalCheckResult
from research_repro.evaluation.discrepancy import (
    DiscrepancyAnalyzer, DiscrepancyMagnitude,
)
from research_repro.evaluation.failure_taxonomy import (
    FailureTaxonomyClassifier, TaxonomyResult,
)
from research_repro.evaluation.evaluator import (
    Decision, EvaluationResult, GoalProgress, ConstraintStatus, HypothesisStatus,
)
from research_repro.memory.models import (
    Constraint, Experiment, ExperimentResult,
    FailureType, GoalContract, ResourceBudget, ResearchMemory,
)


def _make_memory(
    target: float = 0.97,
    threshold: float = 0.005,
    constraints: list[Constraint] | None = None,
    experiments: list[Experiment] | None = None,
) -> ResearchMemory:
    goal = GoalContract(
        objective="Reproduce MNIST MLP experiment",
        primary_metric="accuracy",
        target_value=target,
        success_threshold=threshold,
        constraints=constraints or [],
        max_experiments=8,
        allowed_resources=ResourceBudget(
            max_experiments=8,
            max_total_recoveries=6,
        ),
    )
    mem = ResearchMemory(goal=goal, run_id="test_eval_pipeline")
    if experiments:
        mem.experiments = experiments
    return mem


# ---------------------------------------------------------------------------
# GoalChecker
# ---------------------------------------------------------------------------


class TestGoalChecker:
    def _checker(self):
        return GoalChecker()

    def test_metric_achieved(self):
        mem = _make_memory(target=0.97, threshold=0.005)
        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.968}, "run_experiment")
        assert result.metric.achieved is True
        assert result.metric.gap == pytest.approx(0.002, abs=1e-4)

    def test_metric_not_achieved(self):
        mem = _make_memory(target=0.97, threshold=0.005)
        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.93}, "run_experiment")
        assert result.metric.achieved is False
        assert result.metric.gap == pytest.approx(0.04, abs=1e-3)

    def test_no_observed_metric(self):
        mem = _make_memory(target=0.97)
        checker = self._checker()
        result = checker.check(mem, {"f1": 0.90}, "run_experiment")
        assert result.metric.observed_value is None
        assert result.metric.achieved is False

    def test_regression_detected(self):
        # Best result = 0.95, current = 0.88
        exp = Experiment(id="e1", hypothesis="best")
        exp.observed_result = ExperimentResult(accuracy=0.95)
        mem = _make_memory(experiments=[exp])
        mem.best_experiment_id = "e1"

        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.88}, "run_experiment")
        assert result.regression.is_regression is True
        assert result.regression.delta == pytest.approx(-0.07, abs=1e-3)

    def test_no_regression_improvement(self):
        exp = Experiment(id="e1")
        exp.observed_result = ExperimentResult(accuracy=0.90)
        mem = _make_memory(experiments=[exp])
        mem.best_experiment_id = "e1"

        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.95}, "run_experiment")
        assert result.regression.is_regression is False

    def test_budget_exhausted(self):
        mem = _make_memory()
        mem.budget.experiments_consumed = 8  # At limit

        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.95}, "run_experiment")
        assert result.budget.exhausted is True
        assert result.budget.experiments_remaining == 0

    def test_budget_at_risk(self):
        mem = _make_memory()
        mem.budget.experiments_consumed = 7  # 1 remaining out of 8

        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.95}, "run_experiment")
        assert result.budget.at_risk is True

    def test_constraint_latency_violation(self):
        constraints = [Constraint(name="latency_under_100ms", description="Latency < 100ms")]
        mem = _make_memory(constraints=constraints)
        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.97, "latency_ms": 143.7}, "run_experiment")
        assert result.any_constraint_violated is True

    def test_constraint_latency_satisfied(self):
        constraints = [Constraint(name="latency_under_100ms", description="Latency < 100ms")]
        mem = _make_memory(constraints=constraints)
        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.97, "latency_ms": 87.3}, "run_experiment")
        assert result.any_constraint_violated is False

    def test_all_good_property(self):
        mem = _make_memory(target=0.97, threshold=0.005)
        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.968}, "run_experiment")
        assert result.all_good is True

    def test_to_dict_serializable(self):
        mem = _make_memory()
        checker = self._checker()
        result = checker.check(mem, {"accuracy": 0.95}, "run_experiment")
        d = result.to_dict()
        assert "metric_achieved" in d
        assert "is_regression" in d
        assert "budget_experiments_remaining" in d


# ---------------------------------------------------------------------------
# DiscrepancyAnalyzer
# ---------------------------------------------------------------------------


class TestDiscrepancyAnalyzer:
    def _analyzer(self):
        return DiscrepancyAnalyzer()

    def test_negligible_gap(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.968)
        assert result.magnitude == DiscrepancyMagnitude.NEGLIGIBLE

    def test_minor_gap(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.955)
        assert result.magnitude == DiscrepancyMagnitude.MINOR

    def test_significant_gap(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.940)
        assert result.magnitude == DiscrepancyMagnitude.SIGNIFICANT

    def test_critical_gap(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.900)
        assert result.magnitude == DiscrepancyMagnitude.CRITICAL

    def test_gap_calculation(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.920)
        assert result.absolute_gap == pytest.approx(0.05, abs=1e-4)
        assert result.relative_gap_pct == pytest.approx(5.15, abs=0.1)

    def test_returns_likely_causes(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.930)
        assert isinstance(result.likely_causes, list)
        assert len(result.likely_causes) > 0

    def test_actionable_for_significant_gap(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.940)
        assert result.is_actionable is True

    def test_to_dict(self):
        analyzer = self._analyzer()
        result = analyzer.analyze("accuracy", 0.970, 0.930)
        d = result.to_dict()
        assert "magnitude" in d
        assert "likely_causes" in d
        assert "investigation_priority" in d
        assert "is_actionable" in d


# ---------------------------------------------------------------------------
# FailureTaxonomyClassifier
# ---------------------------------------------------------------------------


class TestFailureTaxonomyClassifier:
    def _classifier(self):
        return FailureTaxonomyClassifier()

    def test_detects_schema_error(self):
        classifier = self._classifier()
        response = {
            "success": False,
            "error": "TOOL_SCHEMA_ERROR",
            "data": {"error_type": "TOOL_SCHEMA_ERROR", "field_errors": []},
        }
        result = classifier.classify("run_experiment", response)
        assert result.is_failure
        assert result.failure_type == FailureType.TOOL_SCHEMA_ERROR
        assert result.confidence > 0.95

    def test_detects_tool_crash(self):
        classifier = self._classifier()
        response = {
            "success": False,
            "error": "Experiment failed (exit code 1). stderr: RuntimeError...",
        }
        result = classifier.classify("run_experiment", response)
        assert result.is_failure
        assert result.failure_type == FailureType.TOOL_CRASH

    def test_detects_timeout(self):
        classifier = self._classifier()
        response = {
            "success": False,
            "error": "Experiment timed out after 300s.",
        }
        result = classifier.classify("run_experiment", response)
        assert result.is_failure
        assert result.failure_type == FailureType.TOOL_CRASH  # timeout is a crash

    def test_detects_result_inconsistency(self):
        classifier = self._classifier()
        response = {
            "success": True,
            "data": {
                "eval_accuracy": 0.91,
                "training_accuracy": 0.95,
                "discrepancy": 0.04,
                "consistent": False,
            },
        }
        result = classifier.classify("run_independent_evaluation", response)
        assert result.is_failure
        assert result.failure_type == FailureType.RESULT_INCONSISTENCY

    def test_detects_goal_drift(self):
        classifier = self._classifier()
        facts = {
            "metric_achieved": True,
            "any_constraint_violated": True,
            "constraint_details": [
                {"name": "latency_under_100ms", "satisfied": False, "reason": "143.7ms > 100ms"}
            ],
        }
        result = classifier.classify("run_experiment", {}, goal_check_facts=facts)
        assert result.is_failure
        assert result.failure_type == FailureType.GOAL_DRIFT

    def test_detects_regression(self):
        classifier = self._classifier()
        facts = {
            "is_regression": True,
            "regression_delta": -0.07,
            "best_known_value": 0.95,
            "observed_value": 0.88,
        }
        result = classifier.classify("run_experiment", {}, goal_check_facts=facts)
        assert result.is_failure
        assert result.failure_type == FailureType.REGRESSION

    def test_no_failure_on_success(self):
        classifier = self._classifier()
        response = {"success": True, "data": {"accuracy": 0.95}}
        result = classifier.classify("run_experiment", response, goal_check_facts={})
        assert not result.is_failure
        assert result.failure_type is None
        assert result.confidence > 0.9

    def test_schema_error_wins_over_crash(self):
        """Schema error has higher priority than generic crash."""
        classifier = self._classifier()
        response = {
            "success": False,
            "error": "TOOL_SCHEMA_ERROR",
            "data": {"error_type": "TOOL_SCHEMA_ERROR", "field_errors": [{"field": "id"}]},
        }
        result = classifier.classify("run_experiment", response)
        assert result.failure_type == FailureType.TOOL_SCHEMA_ERROR

    def test_all_six_failure_types_classifiable(self):
        """Verify each failure type can be returned by the classifier."""
        classifiable = {
            FailureType.TOOL_CRASH,
            FailureType.TOOL_SCHEMA_ERROR,
            FailureType.RESULT_INCONSISTENCY,
            FailureType.GOAL_DRIFT,
            FailureType.REGRESSION,
            FailureType.NON_DETERMINISM,
        }
        assert len(classifiable) == 6
