"""
Experiment Graph — a tree of hypothesis-driven experiments.

Each experiment node has an optional parent, forming a lineage from
the baseline through successive modifications. The graph enables the
agent to reason over research history rather than starting from zero.
"""
from __future__ import annotations

from typing import Optional

from .models import Experiment, ExperimentResult


class ExperimentGraph:
    """
    Wraps the flat experiments list from ResearchMemory as a navigable tree.

    Tree structure:
        BASELINE
           │
        EXP-01 (hypothesis: normalize inputs)
           └── EXP-03 (hypothesis: normalize + cosine LR)
        EXP-02 (hypothesis: different optimizer)
    """

    def __init__(self, experiments: list[Experiment]) -> None:
        self._experiments: dict[str, Experiment] = {
            exp.id: exp for exp in experiments
        }

    def add_node(self, experiment: Experiment) -> None:
        self._experiments[experiment.id] = experiment

    def get(self, experiment_id: str) -> Optional[Experiment]:
        return self._experiments.get(experiment_id)

    def roots(self) -> list[Experiment]:
        """Return experiments with no parent (the baselines)."""
        return [e for e in self._experiments.values() if e.parent_id is None]

    def children(self, parent_id: str) -> list[Experiment]:
        """Return all direct children of an experiment."""
        return [
            e for e in self._experiments.values() if e.parent_id == parent_id
        ]

    def lineage(self, experiment_id: str) -> list[Experiment]:
        """
        Return the chain from root to this experiment (inclusive, root first).
        Used to reconstruct how the agent arrived at a result.
        """
        chain: list[Experiment] = []
        current_id: str | None = experiment_id
        while current_id:
            exp = self._experiments.get(current_id)
            if not exp:
                break
            chain.append(exp)
            current_id = exp.parent_id
        return list(reversed(chain))

    def best(self, metric: str) -> Optional[Experiment]:
        """Return the experiment with the highest value for the given metric."""
        best_exp: Optional[Experiment] = None
        best_val: Optional[float] = None
        for exp in self._experiments.values():
            if exp.observed_result:
                val = exp.observed_result.get_metric(metric)
                if val is not None:
                    if best_val is None or val > best_val:
                        best_val = val
                        best_exp = exp
        return best_exp

    def already_tried(self, hypothesis: str) -> bool:
        """Rough check: has a hypothesis with this text already been run?"""
        return any(
            e.hypothesis.strip().lower() == hypothesis.strip().lower()
            for e in self._experiments.values()
        )

    def to_dict(self) -> dict:
        """Serialize the graph for logging/viewing."""
        return {
            exp_id: {
                "id": exp.id,
                "parent_id": exp.parent_id,
                "hypothesis": exp.hypothesis,
                "execution_status": exp.execution_status.value,
                "evaluation_status": exp.evaluation_status.value,
                "result": (
                    exp.observed_result.model_dump() if exp.observed_result else None
                ),
            }
            for exp_id, exp in self._experiments.items()
        }

    def __len__(self) -> int:
        return len(self._experiments)
