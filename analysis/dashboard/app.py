"""RoboEval Phase 5 interactive dashboard (Plotly/Dash, HF Spaces).

Loads ``DashboardData`` once at boot, keeps it in a client-side
``dcc.Store``, and renders a narrative single-page layout: hero
degradation curves, per-cell failure-mode breakdown, Phase 4 ablation
panel.

Run locally::

    roboeval dashboard run

Deploy: see ``analysis/dashboard/README.md``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
from dash import Dash, html
from dash import dcc as dcc

from roboeval.dashboard.data import load_all
from roboeval.dashboard.figures import (
    build_degradation_curve,
    build_failure_stack,
    build_phase4_ablation,
)
from roboeval.dashboard.models import (
    AblationCondition,
    Cell,
    DashboardData,
    FailureCounts,
    WelchT,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_to_store_dict(data: DashboardData) -> dict[str, Any]:
    """Serialise DashboardData to a JSON-safe dict for ``dcc.Store``."""

    def _convert(value: object) -> object:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                f.name: _convert(getattr(value, f.name))
                for f in dataclasses.fields(value)
            }
        if isinstance(value, list | tuple):
            return [_convert(v) for v in value]
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    result = _convert(data)
    assert isinstance(result, dict)
    return result


def _store_dict_to_data(payload: dict[str, Any]) -> DashboardData:
    """Reverse of :func:`_data_to_store_dict` — rebuild dataclasses from a dict."""

    def _fc(d: dict[str, Any]) -> FailureCounts:
        return FailureCounts(**d)

    cells = tuple(
        Cell(
            cell_id=c["cell_id"],
            axis=c["axis"],
            magnitude=c["magnitude"],
            mean_tsr_custom=c["mean_tsr_custom"],
            std_tsr_custom=c["std_tsr_custom"],
            per_seed_tsr_custom=(
                None
                if c["per_seed_tsr_custom"] is None
                else tuple(c["per_seed_tsr_custom"])
            ),
            mean_tsr=c["mean_tsr"],
            median_tts=c["median_tts"],
            failure_counts=_fc(c["failure_counts"]),
            n_rollouts=c["n_rollouts"],
            run_id=c["run_id"],
        )
        for c in payload["cells"]
    )
    ablation = tuple(
        AblationCondition(
            condition_id=a["condition_id"],
            label=a["label"],
            mean_tsr_custom=a["mean_tsr_custom"],
            std_tsr_custom=a["std_tsr_custom"],
            per_seed_means=tuple(a["per_seed_means"]),
            bootstrap_ci=tuple(a["bootstrap_ci"]),
            failure_counts=_fc(a["failure_counts"]),
            run_id=a["run_id"],
        )
        for a in payload["ablation"]
    )
    welch_tests = tuple(WelchT(**w) for w in payload["welch_tests"])
    return DashboardData(
        cells=cells,
        ablation=ablation,
        welch_tests=welch_tests,
        schema_version=payload["schema_version"],
        generated_at=payload["generated_at"],
    )


_DATA: DashboardData = load_all(repo_root=_repo_root())
_STORE_PAYLOAD = _data_to_store_dict(_DATA)


def _build_layout() -> dbc.Container:
    cell_options = [{"label": c.cell_id, "value": c.cell_id} for c in _DATA.cells]
    return dbc.Container(
        fluid=True,
        className="px-4 py-3",
        children=[
            dcc.Store(id="data-store", data=_STORE_PAYLOAD),
            html.H1("RoboEval — failure modes & residual RL for ACT"),
            html.P(
                "Where state-of-the-art imitation learning breaks under "
                "realistic perturbation, and what residual RL can (and "
                "can't) recover."
            ),
            html.Div(
                [
                    html.A(
                        "GitHub",
                        href="https://github.com/RubenoDechua/roboeval",
                        target="_blank",
                        className="me-3",
                    ),
                    html.A(
                        "PRD",
                        href=(
                            "https://github.com/RubenoDechua/roboeval/blob/"
                            "main/docs/PRD.md"
                        ),
                        target="_blank",
                    ),
                ],
                className="mb-4",
            ),
            html.Hr(),
            html.H2("Degradation curves"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.RadioItems(
                            id="axis-filter",
                            options=[
                                {"label": "Both", "value": "both"},
                                {"label": "Spatial", "value": "spatial"},
                                {"label": "Temporal", "value": "temporal"},
                            ],
                            value="both",
                            inline=True,
                        ),
                        xs=12,
                        md=6,
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            id="metric-toggle",
                            options=[
                                {
                                    "label": "TSR (custom)",
                                    "value": "mean_tsr_custom",
                                },
                                {"label": "TSR (env)", "value": "mean_tsr"},
                                {"label": "TTS", "value": "median_tts"},
                            ],
                            value="mean_tsr_custom",
                            inline=True,
                        ),
                        xs=12,
                        md=6,
                    ),
                ],
                className="mb-2",
            ),
            dcc.Graph(
                id="hero-curve",
                figure=build_degradation_curve(
                    _DATA, metric="mean_tsr_custom", axis_filter="both"
                ),
            ),
            html.Hr(),
            html.H2("Per-cell failure-mode breakdown"),
            dbc.Row(
                dbc.Col(
                    dcc.Dropdown(
                        id="cell-select",
                        options=cell_options,
                        value="y+5cm",
                        clearable=False,
                    ),
                    xs=12,
                    md=4,
                ),
                className="mb-2",
            ),
            dcc.Graph(
                id="failure-stack",
                figure=build_failure_stack(_DATA, cell_id="y+5cm"),
            ),
            html.Hr(),
            html.H2("Phase 4 ablation at +5 cm spatial"),
            dcc.Graph(id="ablation-plot", figure=build_phase4_ablation(_DATA)),
            html.Hr(),
            html.Details(
                [
                    html.Summary("Methods & reproducibility"),
                    html.P(
                        f"3 seeds, 50 rollouts each; data generated at "
                        f"{_DATA.generated_at}."
                    ),
                    html.Ul(
                        [
                            html.Li(
                                f"{c.cell_id}: run_id={c.run_id}, " f"n={c.n_rollouts}"
                            )
                            for c in _DATA.cells
                        ]
                    ),
                ],
                className="mt-3",
            ),
        ],
    )


app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
app.title = "RoboEval — Phase 5 dashboard"
app.layout = _build_layout()
server = app.server
