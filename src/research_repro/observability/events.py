"""
Structured event types for the JSONL observability stream.

Every agent action produces one or more AgentEvents.
The complete JSONL log is the authoritative record of a run.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Every meaningful action in the agent lifecycle produces one of these events."""

    GOAL_SET = "goal_set"
    PLAN_CREATED = "plan_created"
    PHASE_TRANSITION = "phase_transition"

    # ReAct cycle
    REASON = "reason"
    ACTION = "action"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    OBSERVATION = "observation"
    SELF_EVALUATION = "self_evaluation"

    # Failure & recovery
    TOOL_SCHEMA_ERROR = "tool_schema_error"
    FAILURE_DETECTED = "failure_detected"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    REPLAN = "replan"
    NO_PROGRESS = "no_progress"

    # Experiments
    EXPERIMENT_STARTED = "experiment_started"
    EXPERIMENT_COMPLETED = "experiment_completed"

    # Evidence
    EVIDENCE_RECORDED = "evidence_recorded"
    DISCREPANCY_DETECTED = "discrepancy_detected"

    # Budget
    BUDGET_WARNING = "budget_warning"
    BUDGET_EXHAUSTED = "budget_exhausted"

    # Goal achievement
    GOAL_ACHIEVED = "goal_achieved"

    # Terminal
    FINAL_REPORT = "final_report"


class AgentEvent(BaseModel):
    """A single structured event emitted by the agent."""

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    run_id: str
    step: int
    event_type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)
