"""Budgeted dispatcher for typed recovery strategies."""
from __future__ import annotations

from ..memory.models import Failure, Recovery, ResearchMemory
from .base import RecoveryContext, RecoveryOutcome
from .strategies import STRATEGIES


class RecoveryOrchestrator:
    def __init__(self, max_total_recoveries: int = 6, max_recoveries_per_subtask: int = 2):
        self.max_total_recoveries = max_total_recoveries
        self.max_recoveries_per_subtask = max_recoveries_per_subtask

    def recover(self, context: RecoveryContext, subtask_id: str | None = None) -> RecoveryOutcome:
        memory = context.memory
        key = subtask_id or context.failure.experiment_id or context.failure.failure_type.value
        if not memory.budget.can_recover(self.max_total_recoveries) or not memory.budget.can_recover_subtask(key, self.max_recoveries_per_subtask):
            context.failure.recovered = False
            return RecoveryOutcome(False, "budget_exhausted", "Recovery budget exhausted; preserved best known state", unresolved=f"unresolvable subtask: {key}")
        strategy = STRATEGIES.get(context.failure.failure_type)
        if strategy is None:
            return RecoveryOutcome(False, "unknown", "No strategy registered", unresolved="unknown failure type")
        memory.budget.consume_recovery(key)
        outcome = strategy.recover(context)
        recovery = Recovery(
            failure_id=context.failure.id,
            strategy=outcome.strategy,
            action_taken=outcome.action_taken,
            success=outcome.success,
            diagnosis=outcome.diagnosis or outcome.action_taken,
            evidence_used=outcome.evidence_used,
            expected_observation=outcome.expected_observation,
            plan_change=outcome.plan_change,
        )
        memory.recoveries.append(recovery)
        context.failure.recovered = outcome.success
        if outcome.unresolved:
            memory.unresolved_questions.append(outcome.unresolved)
        return outcome
