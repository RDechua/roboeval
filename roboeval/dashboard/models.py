"""Frozen dataclasses for the Phase 5 dashboard data model.

These are JSON-serialisable, pure data containers. No business logic
beyond invariants and a small set of derived properties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FailureCounts:
    """Per-rollout count of each PRD §7.2 failure category."""

    success: int
    grasp_failure: int
    approach_failure: int
    recovery_failure: int
    action_oscillation: int
    timeout: int
    visual_confusion: int
    needs_review: int

    @property
    def total(self) -> int:
        """Return the sum across all failure-mode categories."""
        return (
            self.success
            + self.grasp_failure
            + self.approach_failure
            + self.recovery_failure
            + self.action_oscillation
            + self.timeout
            + self.visual_confusion
            + self.needs_review
        )

    def as_fractions(self) -> dict[str, float]:
        """Return each category as a fraction of total rollouts.

        Raises:
            ValueError: if the bucket is empty (no rollouts to divide by).
        """
        total = self.total
        if total == 0:
            raise ValueError("FailureCounts is empty; no rollouts to normalise.")
        return {
            "success": self.success / total,
            "grasp_failure": self.grasp_failure / total,
            "approach_failure": self.approach_failure / total,
            "recovery_failure": self.recovery_failure / total,
            "action_oscillation": self.action_oscillation / total,
            "timeout": self.timeout / total,
            "visual_confusion": self.visual_confusion / total,
            "needs_review": self.needs_review / total,
        }


@dataclass(frozen=True)
class Cell:
    """One perturbation cell (spatial or temporal) or the nominal anchor."""

    cell_id: str
    axis: Literal["spatial", "temporal", "nominal"]
    magnitude: float
    mean_tsr_custom: float
    std_tsr_custom: float
    per_seed_tsr_custom: tuple[float, ...] | None
    mean_tsr: float | None
    median_tts: float | None
    failure_counts: FailureCounts
    n_rollouts: int
    run_id: str


@dataclass(frozen=True)
class AblationCondition:
    """One arm of the Phase 4 +5 cm ablation (A, B, or C)."""

    condition_id: Literal["A", "B", "C"]
    label: str
    mean_tsr_custom: float
    std_tsr_custom: float
    per_seed_means: tuple[float, float, float]
    bootstrap_ci: tuple[float, float]
    failure_counts: FailureCounts
    run_id: str


@dataclass(frozen=True)
class WelchT:
    """One-sided Welch's t-test result for an ablation arm vs A."""

    arm_id: str
    t_statistic: float
    df: float
    p_one_sided: float


@dataclass(frozen=True)
class DashboardData:
    """Single in-memory bundle the dashboard reads from."""

    cells: tuple[Cell, ...]
    ablation: tuple[AblationCondition, ...]
    welch_tests: tuple[WelchT, ...]
    schema_version: int
    generated_at: str
