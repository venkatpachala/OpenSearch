"""
Discrepancy Analyzer — statistical analysis of result gaps.

When compare_results detects a DISCREPANCY, this module:
  1. Classifies the magnitude (negligible / minor / significant / critical)
  2. Suggests likely causal categories based on the gap size and context
  3. Generates structured diagnostic information for the replanner

This is deterministic — no LLM needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..memory.models import Methodology


class DiscrepancyMagnitude(str, Enum):
    NEGLIGIBLE = "negligible"    # < 0.5% — within noise
    MINOR = "minor"              # 0.5% – 2% — possibly fixable
    SIGNIFICANT = "significant"  # 2% – 5% — likely a systematic issue
    CRITICAL = "critical"        # > 5% — fundamental problem


# Known causal categories with heuristic detection
_KNOWN_CAUSES = [
    {
        "name": "preprocessing_mismatch",
        "description": "Preprocessing pipeline differs from paper (normalization, augmentation)",
        "typical_gap_range": (0.03, 0.08),
        "keywords": ["normalize", "standardize", "augment", "crop", "flip"],
    },
    {
        "name": "lr_schedule_mismatch",
        "description": "Learning rate schedule differs (e.g., step vs cosine)",
        "typical_gap_range": (0.01, 0.04),
        "keywords": ["cosine", "schedule", "warmup", "decay"],
    },
    {
        "name": "optimizer_mismatch",
        "description": "Optimizer or hyperparameters differ from paper",
        "typical_gap_range": (0.01, 0.05),
        "keywords": ["optimizer", "sgd", "adam", "momentum", "weight_decay"],
    },
    {
        "name": "architecture_mismatch",
        "description": "Model architecture differs from paper",
        "typical_gap_range": (0.02, 0.10),
        "keywords": ["architecture", "depth", "width", "layers", "batch_norm"],
    },
    {
        "name": "training_duration",
        "description": "Insufficient training epochs or early stopping",
        "typical_gap_range": (0.01, 0.03),
        "keywords": ["epochs", "iterations", "early_stopping", "convergence"],
    },
    {
        "name": "data_split_mismatch",
        "description": "Different train/test split or evaluation subset",
        "typical_gap_range": (0.005, 0.02),
        "keywords": ["split", "subset", "validation", "test_set"],
    },
    {
        "name": "random_seed",
        "description": "Random initialization variance (non-determinism)",
        "typical_gap_range": (0.001, 0.015),
        "keywords": ["seed", "random", "initialization", "stochastic"],
    },
]


@dataclass
class DiscrepancyAnalysis:
    """Complete analysis of a detected discrepancy."""

    metric: str
    expected: float
    observed: float
    absolute_gap: float
    relative_gap_pct: float
    magnitude: DiscrepancyMagnitude
    likely_causes: list[str]
    cause_descriptions: list[str]
    investigation_priority: list[str]  # Ordered list of what to investigate first
    is_actionable: bool  # False if gap is too large to be fixable realistically

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "expected": self.expected,
            "observed": self.observed,
            "absolute_gap": self.absolute_gap,
            "relative_gap_pct": self.relative_gap_pct,
            "magnitude": self.magnitude.value,
            "likely_causes": self.likely_causes,
            "cause_descriptions": self.cause_descriptions,
            "investigation_priority": self.investigation_priority,
            "is_actionable": self.is_actionable,
        }


class DiscrepancyAnalyzer:
    """
    Analyzes detected discrepancies and suggests investigation priorities.

    Used by the Diagnoser (Phase 6) and as input to the enhanced Evaluator.
    """

    def analyze(
        self,
        metric: str,
        expected: float,
        observed: float,
        methodology: Methodology | None = None,
        prior_experiments: list[dict] | None = None,
    ) -> DiscrepancyAnalysis:
        """
        Analyze a discrepancy and return diagnostic information.

        Args:
            metric:            The metric name (e.g., "accuracy")
            expected:          The paper-reported value
            observed:          Our observed value
            methodology:       Extracted methodology (for context-aware diagnosis)
            prior_experiments: List of prior experiment result summaries
        """
        gap = abs(expected - observed)
        rel_gap = gap / max(expected, 1e-9) * 100
        magnitude = self._classify_magnitude(gap)
        causes = self._identify_likely_causes(gap, methodology)
        is_actionable = magnitude in (
            DiscrepancyMagnitude.MINOR, DiscrepancyMagnitude.SIGNIFICANT
        ) or (magnitude == DiscrepancyMagnitude.CRITICAL and gap < 0.15)

        return DiscrepancyAnalysis(
            metric=metric,
            expected=round(expected, 4),
            observed=round(observed, 4),
            absolute_gap=round(gap, 4),
            relative_gap_pct=round(rel_gap, 2),
            magnitude=magnitude,
            likely_causes=[c["name"] for c in causes],
            cause_descriptions=[c["description"] for c in causes],
            investigation_priority=self._prioritize_investigation(causes, methodology),
            is_actionable=is_actionable,
        )

    def _classify_magnitude(self, gap: float) -> DiscrepancyMagnitude:
        if gap < 0.005:
            return DiscrepancyMagnitude.NEGLIGIBLE
        elif gap < 0.02:
            return DiscrepancyMagnitude.MINOR
        elif gap < 0.05:
            return DiscrepancyMagnitude.SIGNIFICANT
        else:
            return DiscrepancyMagnitude.CRITICAL

    def _identify_likely_causes(
        self, gap: float, methodology: Methodology | None
    ) -> list[dict]:
        """Return causes whose typical gap range overlaps with observed gap."""
        likely = []
        for cause in _KNOWN_CAUSES:
            low, high = cause["typical_gap_range"]
            # Include if gap falls within typical range ± 50%
            if low * 0.5 <= gap <= high * 1.5:
                likely.append(cause)

        # If methodology is available, boost causes matching its keywords
        if methodology:
            method_str = str(methodology.model_dump()).lower()
            for cause in likely:
                if any(kw in method_str for kw in cause["keywords"]):
                    # Move to front
                    likely.remove(cause)
                    likely.insert(0, cause)

        return likely[:4]  # Return top 4

    def _prioritize_investigation(
        self, causes: list[dict], methodology: Methodology | None
    ) -> list[str]:
        """Ordered list of what to investigate first."""
        priority = []
        for cause in causes:
            priority.append(
                f"Investigate {cause['name']}: {cause['description']}"
            )
        if not priority:
            priority = [
                "Inspect paper's preprocessing section for normalization details",
                "Check paper's training details for LR schedule",
                "Verify optimizer configuration matches paper",
            ]
        return priority
