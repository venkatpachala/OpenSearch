"""
Tests for the tool system — validation, schema errors, and registry.
"""
import pytest
from pydantic import ValidationError

from research_repro.tools.base import ToolRegistry, ToolResponse, ToolSchemaError
from research_repro.tools.stub import (
    RunExperimentTool,
    SearchLiteratureTool,
    build_stub_registry,
)


# ---------------------------------------------------------------------------
# Registry basics
# ---------------------------------------------------------------------------


def test_registry_contains_all_stub_tools():
    registry = build_stub_registry()
    expected = {
        "search_literature",
        "retrieve_paper",
        "extract_methodology",
        "run_experiment",
        "collect_metrics",
        "compare_results",
        "inspect_methodology",
    }
    available = {t["name"] for t in registry.list_available()}
    assert expected.issubset(available)


def test_registry_unknown_tool_raises():
    registry = build_stub_registry()
    with pytest.raises(KeyError, match="nonexistent_tool"):
        registry.execute("nonexistent_tool", {})


# ---------------------------------------------------------------------------
# Successful tool execution
# ---------------------------------------------------------------------------


def test_search_literature_returns_papers():
    registry = build_stub_registry()
    resp = registry.execute("search_literature", {"query": "CIFAR-10 ResNet", "max_results": 3})
    assert resp.success is True
    assert "papers" in resp.data
    assert len(resp.data["papers"]) == 3


def test_retrieve_paper_returns_sections():
    registry = build_stub_registry()
    resp = registry.execute("retrieve_paper", {"paper_id": "paper_1"})
    assert resp.success is True
    assert "sections" in resp.data
    assert "methodology" in resp.data["sections"]


def test_extract_methodology_returns_hyperparameters():
    registry = build_stub_registry()
    resp = registry.execute(
        "extract_methodology", {"paper_id": "paper_1", "section": "experimental_setup"}
    )
    assert resp.success is True
    assert resp.data["hyperparameters"]["optimizer"] == "SGD"
    assert resp.data["reported_metrics"]["accuracy"] == pytest.approx(0.942)


def test_run_experiment_baseline_below_target():
    """Baseline should return ~0.867, deliberately below reported 0.942."""
    registry = build_stub_registry()
    resp = registry.execute(
        "run_experiment",
        {"experiment_id": "exp_001", "seed": 42},
    )
    assert resp.success is True
    acc = resp.data["accuracy"]
    assert 0.8 < acc < 0.90, f"Expected baseline ~0.867, got {acc}"


def test_run_experiment_with_normalize_fix_improves():
    """Passing 'normalize' in parameters should improve accuracy."""
    registry = build_stub_registry()
    resp = registry.execute(
        "run_experiment",
        {
            "experiment_id": "exp_002",
            "parameters": {"preprocessing": "normalize"},
            "seed": 42,
        },
    )
    assert resp.success is True
    assert resp.data["accuracy"] > 0.867


def test_compare_results_discrepancy():
    registry = build_stub_registry()
    resp = registry.execute(
        "compare_results",
        {
            "experiment_id": "exp_001",
            "reported_accuracy": 0.942,
            "observed_accuracy": 0.867,
        },
    )
    assert resp.success is True
    assert resp.data["status"] == "DISCREPANCY"
    assert resp.data["absolute_difference"] > 0.005


def test_compare_results_reproduced():
    registry = build_stub_registry()
    resp = registry.execute(
        "compare_results",
        {
            "experiment_id": "exp_006",
            "reported_accuracy": 0.942,
            "observed_accuracy": 0.941,
            "threshold": 0.005,
        },
    )
    assert resp.success is True
    assert resp.data["status"] == "REPRODUCED"


def test_tool_response_ok_helper():
    resp = ToolResponse.ok(value=42, name="test")
    assert resp.success is True
    assert resp.data == {"value": 42, "name": "test"}
    assert resp.error is None


def test_tool_response_fail_helper():
    resp = ToolResponse.fail("something went wrong")
    assert resp.success is False
    assert resp.error == "something went wrong"


# ---------------------------------------------------------------------------
# Schema error handling (Failure Type 2)
# ---------------------------------------------------------------------------


def test_schema_error_on_wrong_type():
    """experiment_id must be str, not int."""
    registry = build_stub_registry()
    with pytest.raises(ToolSchemaError) as exc_info:
        registry.execute(
            "run_experiment",
            {
                "experiment_id": 123,          # int instead of str
                "timeout_seconds": "five",     # str instead of int
            },
        )
    err = exc_info.value
    assert err.tool_name == "run_experiment"
    assert len(err.field_errors) >= 1


def test_schema_error_feedback_structure():
    """to_feedback() must return structured data the agent can use to repair args."""
    registry = build_stub_registry()
    try:
        registry.execute("run_experiment", {"experiment_id": 999})
    except ToolSchemaError as e:
        feedback = e.to_feedback()
        assert feedback["error_type"] == "TOOL_SCHEMA_ERROR"
        assert feedback["tool_name"] == "run_experiment"
        assert isinstance(feedback["field_errors"], list)
        assert all("field" in fe and "message" in fe for fe in feedback["field_errors"])


def test_schema_error_missing_required_field():
    """run_experiment requires experiment_id."""
    registry = build_stub_registry()
    with pytest.raises(ToolSchemaError):
        registry.execute("run_experiment", {})  # missing experiment_id


def test_tool_schema_error_inherits_exception():
    from pydantic import ValidationError as PydanticValidationError
    try:
        from research_repro.tools.stub import RunExperimentRequest
        RunExperimentRequest(**{"experiment_id": 99})
    except Exception as e:
        schema_err = ToolSchemaError("run_experiment", e)
        assert isinstance(schema_err, Exception)
        assert "run_experiment" in str(schema_err)
