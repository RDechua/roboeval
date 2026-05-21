"""Frozen-dataclass invariants for the dashboard data model."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from roboeval.dashboard.models import (
    AblationCondition,
    Cell,
    DashboardData,
    FailureCounts,
    WelchT,
)


def _make_counts(success: int = 30, recovery: int = 90) -> FailureCounts:
    return FailureCounts(
        success=success,
        grasp_failure=0,
        approach_failure=0,
        recovery_failure=recovery,
        action_oscillation=0,
        timeout=0,
        visual_confusion=0,
        needs_review=150 - success - recovery,
    )


def test_failure_counts_total_sums_all_categories() -> None:
    counts = _make_counts(success=30, recovery=90)
    assert counts.total == 150


def test_failure_counts_as_fractions_sums_to_one() -> None:
    counts = _make_counts(success=30, recovery=90)
    fractions = counts.as_fractions()
    assert pytest.approx(sum(fractions.values()), abs=1e-9) == 1.0


def test_failure_counts_as_fractions_zero_total_raises() -> None:
    empty = FailureCounts(0, 0, 0, 0, 0, 0, 0, 0)
    with pytest.raises(ValueError, match="empty"):
        empty.as_fractions()


def test_cell_is_frozen() -> None:
    cell = Cell(
        cell_id="y+5cm",
        axis="spatial",
        magnitude=0.05,
        mean_tsr_custom=0.307,
        std_tsr_custom=0.019,
        per_seed_tsr_custom=None,
        mean_tsr=None,
        median_tts=None,
        failure_counts=_make_counts(46, 89),
        n_rollouts=150,
        run_id="w6k2wole",
    )
    with pytest.raises(FrozenInstanceError):
        cell.mean_tsr_custom = 0.0  # type: ignore[misc]


def test_ablation_condition_per_seed_tuple_length() -> None:
    cond = AblationCondition(
        condition_id="A",
        label="Frozen base only",
        mean_tsr_custom=0.32,
        std_tsr_custom=0.059,
        per_seed_means=(0.26, 0.4, 0.3),
        bootstrap_ci=(0.26, 0.4),
        failure_counts=_make_counts(48, 89),
        run_id="w6k2wole",
    )
    assert len(cond.per_seed_means) == 3


def test_welch_t_fields_present() -> None:
    w = WelchT(arm_id="B", t_statistic=-2.95, df=2.7, p_one_sided=0.034)
    assert w.arm_id == "B"
    assert w.p_one_sided < 0.05


def test_dashboard_data_aggregates_all_pieces() -> None:
    cell = Cell(
        cell_id="nominal",
        axis="nominal",
        magnitude=0.0,
        mean_tsr_custom=0.8,
        std_tsr_custom=0.057,
        per_seed_tsr_custom=None,
        mean_tsr=None,
        median_tts=None,
        failure_counts=_make_counts(120, 0),
        n_rollouts=150,
        run_id="nominal-anchor",
    )
    cond = AblationCondition(
        condition_id="A",
        label="Frozen base only",
        mean_tsr_custom=0.32,
        std_tsr_custom=0.059,
        per_seed_means=(0.26, 0.4, 0.3),
        bootstrap_ci=(0.26, 0.4),
        failure_counts=_make_counts(48, 89),
        run_id="w6k2wole",
    )
    welch = WelchT(arm_id="B", t_statistic=-2.95, df=2.7, p_one_sided=0.034)
    data = DashboardData(
        cells=(cell,),
        ablation=(cond,),
        welch_tests=(welch,),
        schema_version=1,
        generated_at="2026-05-21T00:00:00Z",
    )
    assert data.schema_version == 1
    assert data.cells[0].cell_id == "nominal"
