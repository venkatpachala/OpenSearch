"""
Agent Phase State Machine.

Defines the valid lifecycle states and enforces legal transitions.
The agent MUST move through phases via this state machine — arbitrary
phase jumps are rejected with InvalidTransitionError.
"""
from __future__ import annotations

from enum import Enum


class AgentPhase(str, Enum):
    INITIALIZING = "initializing"
    LITERATURE_DISCOVERY = "literature_discovery"
    PAPER_SELECTION = "paper_selection"
    METHOD_EXTRACTION = "method_extraction"
    EXPERIMENT_PLANNING = "experiment_planning"
    RUNNING_EXPERIMENT = "running_experiment"
    SELF_EVALUATION = "self_evaluation"
    DIAGNOSING = "diagnosing"
    RECOVERING = "recovering"
    REPLANNING = "replanning"
    GENERATING_REPORT = "generating_report"
    COMPLETED = "completed"
    FAILED = "failed"


# Valid (from → [to, ...]) transition table.
# Any (from, to) pair NOT listed here is illegal.
TRANSITIONS: dict[AgentPhase, list[AgentPhase]] = {
    AgentPhase.INITIALIZING: [
        AgentPhase.LITERATURE_DISCOVERY,
    ],
    AgentPhase.LITERATURE_DISCOVERY: [
        AgentPhase.PAPER_SELECTION,
        AgentPhase.FAILED,
    ],
    AgentPhase.PAPER_SELECTION: [
        AgentPhase.METHOD_EXTRACTION,
        AgentPhase.FAILED,
    ],
    AgentPhase.METHOD_EXTRACTION: [
        AgentPhase.EXPERIMENT_PLANNING,
        AgentPhase.FAILED,
    ],
    AgentPhase.EXPERIMENT_PLANNING: [
        AgentPhase.RUNNING_EXPERIMENT,
        AgentPhase.FAILED,
    ],
    AgentPhase.RUNNING_EXPERIMENT: [
        AgentPhase.SELF_EVALUATION,
        AgentPhase.DIAGNOSING,
        AgentPhase.FAILED,
    ],
    AgentPhase.SELF_EVALUATION: [
        AgentPhase.RUNNING_EXPERIMENT,       # continue to next experiment
        AgentPhase.EXPERIMENT_PLANNING,      # replan experiment
        AgentPhase.DIAGNOSING,               # failure detected
        AgentPhase.GENERATING_REPORT,        # goal achieved or budget exhausted
    ],
    AgentPhase.DIAGNOSING: [
        AgentPhase.RECOVERING,
        AgentPhase.GENERATING_REPORT,        # unresolvable — give up gracefully
    ],
    AgentPhase.RECOVERING: [
        AgentPhase.REPLANNING,
        AgentPhase.GENERATING_REPORT,        # recovery budget exhausted
    ],
    AgentPhase.REPLANNING: [
        AgentPhase.RUNNING_EXPERIMENT,
        AgentPhase.EXPERIMENT_PLANNING,
        AgentPhase.GENERATING_REPORT,
    ],
    AgentPhase.GENERATING_REPORT: [
        AgentPhase.COMPLETED,
    ],
    AgentPhase.COMPLETED: [],
    AgentPhase.FAILED: [],
}

TERMINAL_PHASES = {AgentPhase.COMPLETED, AgentPhase.FAILED}


class InvalidTransitionError(Exception):
    def __init__(self, from_phase: AgentPhase, to_phase: AgentPhase) -> None:
        valid = [p.value for p in TRANSITIONS.get(from_phase, [])]
        super().__init__(
            f"Invalid phase transition: {from_phase.value!r} → {to_phase.value!r}. "
            f"Valid targets from {from_phase.value!r}: {valid}"
        )
        self.from_phase = from_phase
        self.to_phase = to_phase


class StateMachine:
    """
    Enforces valid lifecycle transitions for the agent.

    Usage:
        sm = StateMachine()
        sm.transition(AgentPhase.LITERATURE_DISCOVERY)  # OK
        sm.transition(AgentPhase.COMPLETED)              # raises InvalidTransitionError
    """

    def __init__(self, initial: AgentPhase = AgentPhase.INITIALIZING) -> None:
        self.current = initial

    def transition(self, to: AgentPhase) -> None:
        """Move to a new phase. Raises InvalidTransitionError if illegal."""
        valid = TRANSITIONS.get(self.current, [])
        if to not in valid:
            raise InvalidTransitionError(self.current, to)
        self.current = to

    def can_transition(self, to: AgentPhase) -> bool:
        return to in TRANSITIONS.get(self.current, [])

    @property
    def is_terminal(self) -> bool:
        return self.current in TERMINAL_PHASES
