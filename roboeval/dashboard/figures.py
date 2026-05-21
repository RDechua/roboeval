"""Plotly figure builders for the Phase 5 dashboard.

Each function takes typed dashboard data and returns a
``plotly.graph_objects.Figure``. The functions are pure and stateless;
all PRD §9.1 acceptance properties (axis labels, units, titles) are
locked in by tests in ``tests/dashboard/test_figures.py``.
"""

from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from roboeval.dashboard.models import DashboardData

Metric = Literal["mean_tsr_custom", "mean_tsr", "median_tts"]
AxisFilter = Literal["both", "spatial", "temporal"]

_METRIC_LABELS: dict[str, str] = {
    "mean_tsr_custom": "Mean TSR (custom geometric criterion)",
    "mean_tsr": "Mean TSR (env reward)",
    "median_tts": "Median time-to-success (steps)",
}

_PRIMARY_COLOR = "#2E86AB"
_RIBBON_COLOR = "rgba(46, 134, 171, 0.18)"


def _spatial_series(
    data: DashboardData, metric: Metric
) -> tuple[list[float], list[float], list[float]]:
    cells = sorted(
        [c for c in data.cells if c.axis in ("spatial", "nominal")],
        key=lambda c: c.magnitude,
    )
    xs = [c.magnitude * 100.0 for c in cells]  # m → cm for display
    ys = [c.mean_tsr_custom for c in cells]
    sigmas = [c.std_tsr_custom for c in cells]
    if metric == "mean_tsr":
        ys = [
            c.mean_tsr if c.mean_tsr is not None else c.mean_tsr_custom for c in cells
        ]
    if metric == "median_tts":
        ys = [c.median_tts if c.median_tts is not None else 0.0 for c in cells]
    return xs, ys, sigmas


def _temporal_series(
    data: DashboardData, metric: Metric
) -> tuple[list[float], list[float], list[float]]:
    cells = sorted(
        [c for c in data.cells if c.axis in ("temporal", "nominal")],
        key=lambda c: c.magnitude,
    )
    xs = [c.magnitude for c in cells]
    ys = [c.mean_tsr_custom for c in cells]
    sigmas = [c.std_tsr_custom for c in cells]
    if metric == "mean_tsr":
        ys = [
            c.mean_tsr if c.mean_tsr is not None else c.mean_tsr_custom for c in cells
        ]
    if metric == "median_tts":
        ys = [c.median_tts if c.median_tts is not None else 0.0 for c in cells]
    return xs, ys, sigmas


def _mean_and_ribbon_traces(
    xs: list[float],
    ys: list[float],
    sigmas: list[float],
    *,
    name: str,
) -> list[go.Scatter]:
    upper = [y + s for y, s in zip(ys, sigmas, strict=True)]
    lower = [y - s for y, s in zip(ys, sigmas, strict=True)]
    return [
        go.Scatter(
            x=xs + xs[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor=_RIBBON_COLOR,
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            showlegend=False,
            name=f"{name} ±σ",  # noqa: RUF001 - sigma is the standard symbol
        ),
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line={"color": _PRIMARY_COLOR, "width": 3},
            marker={"size": 8},
            name=name,
        ),
    ]


def build_degradation_curve(
    data: DashboardData,
    *,
    metric: Metric,
    axis_filter: AxisFilter,
) -> go.Figure:
    """Hero figure: TSR vs perturbation magnitude, side-by-side panels."""
    show_spatial = axis_filter in ("both", "spatial")
    show_temporal = axis_filter in ("both", "temporal")

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        subplot_titles=("Spatial perturbation", "Temporal delay"),
        horizontal_spacing=0.08,
    )

    if show_spatial:
        xs, ys, sigmas = _spatial_series(data, metric)
        for tr in _mean_and_ribbon_traces(xs, ys, sigmas, name="spatial"):
            fig.add_trace(tr, row=1, col=1)
    if show_temporal:
        xs, ys, sigmas = _temporal_series(data, metric)
        for tr in _mean_and_ribbon_traces(xs, ys, sigmas, name="temporal"):
            fig.add_trace(tr, row=1, col=2)

    fig.update_xaxes(
        title_text="Perturbation magnitude (cm)", row=1, col=1, zeroline=True
    )
    fig.update_xaxes(title_text="Action delay (steps)", row=1, col=2, zeroline=True)
    fig.update_yaxes(
        title_text=_METRIC_LABELS[metric],
        range=[0, 1] if metric != "median_tts" else None,
        row=1,
        col=1,
    )
    fig.update_layout(
        title="Spatial vs temporal degradation — ACT on AlohaTransferCube",
        template="plotly_white",
        showlegend=False,
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
        height=420,
    )
    return fig


_FAILURE_MODE_ORDER: tuple[tuple[str, str, str], ...] = (
    # (key, display name, color)
    ("success", "Success", "#2CA02C"),
    ("grasp_failure", "Grasp", "#D62728"),
    ("approach_failure", "Approach", "#FF7F0E"),
    ("recovery_failure", "Recovery", "#1F77B4"),
    ("action_oscillation", "Oscillation", "#9467BD"),
    ("timeout", "Timeout", "#8C564B"),
    ("visual_confusion", "Visual confusion", "#E377C2"),
    ("needs_review", "Needs review", "#7F7F7F"),
)


def build_failure_stack(
    data: DashboardData,
    *,
    cell_id: str,
) -> go.Figure:
    """Stacked bar of failure-mode fractions for one perturbation cell."""
    cell = next((c for c in data.cells if c.cell_id == cell_id), None)
    if cell is None:
        raise ValueError(f"unknown cell {cell_id!r}")
    fractions = cell.failure_counts.as_fractions()

    fig = go.Figure()
    for key, display, color in _FAILURE_MODE_ORDER:
        value_pct = fractions[key] * 100.0
        rollout_count = int(round(value_pct * cell.n_rollouts / 100))
        fig.add_trace(
            go.Bar(
                x=[cell.cell_id],
                y=[value_pct],
                name=display,
                marker_color=color,
                hovertemplate=(
                    f"{display}: %{{y:.1f}}% "
                    f"({rollout_count} rollouts)<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        title=(
            f"Failure-mode breakdown — {cell.cell_id} "
            f"({cell.n_rollouts} rollouts, 3 seeds)"
        ),
        template="plotly_white",
        yaxis_title="Fraction of rollouts (%)",
        xaxis_title="Cell",
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
        height=380,
        legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "x": 1.02},
    )
    fig.update_yaxes(range=[0, 100])
    return fig
