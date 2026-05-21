"""Tests for the Plotly figure builders.

Each test verifies a PRD §9.1 acceptance criterion mechanically:
axis labels present, units in titles, filter behavior changes
trace counts. No headless browser; we inspect Plotly Figure objects
directly.
"""

from __future__ import annotations

import pytest

plotly = pytest.importorskip("plotly")

from roboeval.dashboard.figures import build_degradation_curve  # noqa: E402
from roboeval.dashboard.models import (  # noqa: E402
    Cell,
    DashboardData,
    FailureCounts,
)


def _empty_counts() -> FailureCounts:
    return FailureCounts(0, 0, 0, 0, 0, 0, 0, 0)


def _make_cells() -> tuple[Cell, ...]:
    return (
        Cell(
            "y-5cm",
            "spatial",
            -0.05,
            0.127,
            0.009,
            None,
            None,
            None,
            FailureCounts(19, 0, 7, 121, 0, 0, 0, 3),
            150,
            "rid1",
        ),
        Cell(
            "y-3cm",
            "spatial",
            -0.03,
            0.553,
            0.025,
            None,
            None,
            None,
            FailureCounts(83, 0, 0, 63, 0, 0, 0, 4),
            150,
            "rid2",
        ),
        Cell(
            "y-1cm",
            "spatial",
            -0.01,
            0.827,
            0.034,
            None,
            None,
            None,
            FailureCounts(124, 0, 0, 26, 0, 0, 0, 0),
            150,
            "rid3",
        ),
        Cell(
            "nominal",
            "nominal",
            0.0,
            0.800,
            0.057,
            None,
            None,
            None,
            FailureCounts(120, 0, 0, 0, 0, 28, 0, 2),
            150,
            "rid_nominal",
        ),
        Cell(
            "y+1cm",
            "spatial",
            0.01,
            0.720,
            0.102,
            None,
            None,
            None,
            FailureCounts(108, 0, 0, 37, 0, 0, 0, 5),
            150,
            "rid4",
        ),
        Cell(
            "y+3cm",
            "spatial",
            0.03,
            0.553,
            0.041,
            None,
            None,
            None,
            FailureCounts(83, 0, 1, 56, 0, 0, 0, 10),
            150,
            "rid5",
        ),
        Cell(
            "y+5cm",
            "spatial",
            0.05,
            0.307,
            0.019,
            None,
            None,
            None,
            FailureCounts(46, 0, 1, 89, 0, 0, 0, 13),
            150,
            "rid6",
        ),
        Cell(
            "delay-1step",
            "temporal",
            1.0,
            0.753,
            0.050,
            None,
            None,
            None,
            FailureCounts(113, 0, 0, 33, 0, 0, 0, 4),
            150,
            "rid7",
        ),
        Cell(
            "delay-3step",
            "temporal",
            3.0,
            0.767,
            0.068,
            None,
            None,
            None,
            FailureCounts(115, 0, 0, 32, 0, 0, 0, 3),
            150,
            "rid8",
        ),
        Cell(
            "delay-5step",
            "temporal",
            5.0,
            0.687,
            0.066,
            None,
            None,
            None,
            FailureCounts(103, 0, 0, 45, 0, 0, 0, 2),
            150,
            "rid9",
        ),
    )


def _make_data() -> DashboardData:
    return DashboardData(
        cells=_make_cells(),
        ablation=(),
        welch_tests=(),
        schema_version=1,
        generated_at="2026-05-21T00:00:00Z",
    )


def test_degradation_curve_has_axis_labels_and_title() -> None:
    fig = build_degradation_curve(
        _make_data(), metric="mean_tsr_custom", axis_filter="both"
    )
    layout = fig.layout
    assert layout.title is not None and layout.title.text
    assert layout.xaxis.title.text is not None
    assert layout.xaxis2.title.text is not None
    assert layout.yaxis.title.text is not None
    assert "cm" in layout.xaxis.title.text.lower()
    assert "step" in layout.xaxis2.title.text.lower()


def test_degradation_curve_axis_filter_spatial_hides_temporal() -> None:
    fig = build_degradation_curve(
        _make_data(), metric="mean_tsr_custom", axis_filter="spatial"
    )
    visible_traces = [tr for tr in fig.data if tr.visible is not False]
    assert len(visible_traces) == 2


def test_degradation_curve_metric_toggle_changes_y_title() -> None:
    fig_custom = build_degradation_curve(
        _make_data(), metric="mean_tsr_custom", axis_filter="both"
    )
    fig_env = build_degradation_curve(
        _make_data(), metric="mean_tsr", axis_filter="both"
    )
    assert fig_custom.layout.yaxis.title.text != fig_env.layout.yaxis.title.text


from roboeval.dashboard.figures import build_failure_stack  # noqa: E402


def test_failure_stack_has_named_failure_categories() -> None:
    data = _make_data()
    fig = build_failure_stack(data, cell_id="y+5cm")
    trace_names = {tr.name for tr in fig.data}
    assert "Success" in trace_names
    assert "Recovery" in trace_names


def test_failure_stack_axis_labels_present() -> None:
    fig = build_failure_stack(_make_data(), cell_id="y-5cm")
    assert fig.layout.yaxis.title.text is not None
    assert "%" in fig.layout.yaxis.title.text or "fraction" in (
        fig.layout.yaxis.title.text.lower()
    )
    assert fig.layout.title is not None
    assert fig.layout.title.text is not None
    assert "y-5cm" in fig.layout.title.text


def test_failure_stack_unknown_cell_raises() -> None:
    with pytest.raises(ValueError, match="unknown cell"):
        build_failure_stack(_make_data(), cell_id="not-a-cell")


from roboeval.dashboard.figures import build_phase4_ablation  # noqa: E402
from roboeval.dashboard.models import AblationCondition, WelchT  # noqa: E402


def _make_ablation_data() -> DashboardData:
    cells = _make_cells()
    a = AblationCondition(
        condition_id="A",
        label="Frozen base only",
        mean_tsr_custom=0.32,
        std_tsr_custom=0.059,
        per_seed_means=(0.26, 0.4, 0.3),
        bootstrap_ci=(0.26, 0.4),
        failure_counts=FailureCounts(48, 1, 1, 89, 0, 0, 0, 11),
        run_id="w6k2wole",
    )
    b = AblationCondition(
        condition_id="B",
        label="Residual RL, sparse",
        mean_tsr_custom=0.187,
        std_tsr_custom=0.025,
        per_seed_means=(0.22, 0.18, 0.16),
        bootstrap_ci=(0.16, 0.22),
        failure_counts=FailureCounts(28, 1, 8, 106, 0, 0, 0, 7),
        run_id="o6ukyo53",
    )
    c = AblationCondition(
        condition_id="C",
        label="Residual RL, shaped",
        mean_tsr_custom=0.213,
        std_tsr_custom=0.050,
        per_seed_means=(0.28, 0.2, 0.16),
        bootstrap_ci=(0.16, 0.28),
        failure_counts=FailureCounts(32, 0, 4, 109, 0, 0, 0, 5),
        run_id="43czuigy",
    )
    return DashboardData(
        cells=cells,
        ablation=(a, b, c),
        welch_tests=(
            WelchT("B", -2.95, 2.7, 0.034),
            WelchT("C", -1.95, 3.9, 0.062),
        ),
        schema_version=1,
        generated_at="2026-05-21T00:00:00Z",
    )


def test_phase4_ablation_has_three_x_categories() -> None:
    fig = build_phase4_ablation(_make_ablation_data())
    for tr in fig.data:
        assert list(tr.x) == ["A", "B", "C"]


def test_phase4_ablation_title_mentions_plus_5cm() -> None:
    fig = build_phase4_ablation(_make_ablation_data())
    assert fig.layout.title is not None
    title = fig.layout.title.text
    assert title is not None
    assert "+5" in title or "+5cm" in title
