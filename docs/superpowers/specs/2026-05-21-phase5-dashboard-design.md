# Phase 5 — Interactive Dashboard (Design)

**Status:** implemented 2026-05-21 · **Owner:** Rubeno Dechua · **Targets:** PRD §9 (Interactive Dashboard deliverable), PRD §9.1 (quality checklist), PRD §5.3 (`analysis/` layout).

**Update 2026-05-21 (post-implementation):** `data/headline.json` was bumped from schema v1 to v2 to make it self-contained. The original design had the runtime dashboard read from `data/headline.json` + `docs/figures/phase4_ablation.json` + 3 `outputs/.../eval_results_*.json` + 3 `data/taxonomy/auto_labels_*.json`. The latter two paths are gitignored, so the tests failed in CI even though they passed locally. Schema v2 bakes the Phase 4 ablation block (per-condition stats, per-seed means, bootstrap CI, failure counts) and the Welch's t-tests into `data/headline.json` at *build* time. The runtime now reads exactly one tracked file. Build script: `scripts/build_headline_json.py`. Runtime loader: `roboeval.dashboard.data.load_dashboard_data`.

## 1. Goal

Ship a hosted, mobile-responsive Plotly/Dash web app that tells the v1.0 RoboEval story end-to-end: spatial + temporal degradation curves, per-cell failure-mode breakdown, and the Phase 4 ablation result. The app must satisfy PRD §9.1 acceptance:

- Loads in <3 s warm.
- Mobile-responsive.
- All plots have axis labels, units, and titles.
- Failure-mode filter works without page reload.

**Primary audience:** product / applied ML hiring teams who land on the page from the GitHub README or a job-application link. Story-first, drill-down second.

**Non-goals (v1.0):**

- Per-rollout trajectory replay (deferred to v1.1).
- Multi-policy comparison (only ACT in v1.0 per PRD non-goal).
- Live W&B fetch on page load (data is pre-baked at build time).

## 2. Story arc

Single-page, vertical scroll, top-to-bottom:

1. **Header strip** — title, one-sentence tagline, links to GitHub / blog / PRD.
2. **Hero (above the fold)** — two side-by-side subplots: spatial degradation curve (cells in cm) and temporal degradation curve (cells in steps). Shared y-axis. ±σ ribbons. Caption summarizes the cross-axis elasticity finding ("spatial brittle, temporal robust"). Filters: perturbation axis (Both / Spatial / Temporal) + metric toggle (TSR_custom / TSR_env / TTS).
3. **Failure-mode breakdown** — stacked-bar of failure-mode fractions for one cell, driven by a cell-selector dropdown. Legend click highlights one mode (PRD-required filter).
4. **Phase 4 ablation panel** — interactive Plotly version of the existing `phase4_ablation_failure_distribution.png` plus a Welch's t-test table (B vs A, C vs A, one-sided).
5. **Methods & reproducibility** (collapsed) — config paths, seeds, run IDs, git SHA, schema_version, generated_at.
6. **Footer** — author, MIT license, repo link.

Mobile (<768 px): hero subplots stack vertically; Welch table becomes a definition list.

## 3. Architecture

Split-package layout. Pure logic in the `roboeval/` package (mypy --strict, fully unit-tested); the Dash app skeleton lives in `analysis/dashboard/` per PRD §5.3.

```
roboeval/dashboard/
├── __init__.py
├── models.py      # frozen dataclasses (Cell, FailureCounts, AblationCondition, WelchT, DashboardData)
├── data.py        # load_headline_json, load_phase4_ablation, load_phase4_eval_results, load_all
└── figures.py     # build_degradation_curve, build_failure_stack, build_phase4_ablation

analysis/dashboard/
├── app.py         # ~80 LOC: Dash layout, callbacks, server entry
├── Dockerfile     # python:3.11-slim, gunicorn on :7860 for HF Spaces
├── requirements.txt  # pinned subset of pyproject.toml deps
└── README.md      # HF Space frontmatter + "first visit ~30s while container wakes"

data/
└── headline.json  # hand-written from docs/STATE.md (10 Phase 3 cells)

tests/dashboard/
├── test_data.py
├── test_models.py
├── test_figures.py
└── test_callbacks.py
```

`roboeval/cli.py` gains a `dashboard` subcommand with `run` (launches the Dash dev server) and `build` (validates that all required data files load cleanly).

**Boundary discipline:** `roboeval/dashboard/` does not import `dash`. Only `analysis/dashboard/app.py` does. This keeps the figure logic hot-reloadable in a notebook and keeps mypy --strict honest on the package.

## 4. Data model

All frozen dataclasses (Python `@dataclass(frozen=True)`). No Pydantic — mypy --strict prefers stdlib, JSON shape is small and stable.

```python
@dataclass(frozen=True)
class FailureCounts:
    success: int
    grasp_failure: int
    approach_failure: int
    recovery_failure: int
    action_oscillation: int
    timeout: int
    visual_confusion: int
    needs_review: int

@dataclass(frozen=True)
class Cell:
    cell_id: str                                # "y-5cm", "y+5cm", "delay-5step", "nominal"
    axis: Literal["spatial", "temporal", "nominal"]
    magnitude: float                            # cm for spatial, steps for temporal
    mean_tsr_custom: float
    std_tsr_custom: float
    per_seed_tsr_custom: tuple[float, ...] | None   # None for cells without per-seed data
    mean_tsr: float | None
    median_tts: float | None
    failure_counts: FailureCounts
    n_rollouts: int
    run_id: str

@dataclass(frozen=True)
class AblationCondition:
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
    arm_id: str
    t_statistic: float
    df: float
    p_one_sided: float

@dataclass(frozen=True)
class DashboardData:
    cells: tuple[Cell, ...]
    ablation: tuple[AblationCondition, ...]
    welch_tests: tuple[WelchT, ...]
    schema_version: int
    generated_at: str
```

**Data sources:**

- Phase 3 cells → `data/headline.json` (new, hand-written from `docs/STATE.md`).
- Phase 4 ablation (mean ± σ, per-seed, bootstrap CI, Welch's t) → existing `docs/figures/phase4_ablation.json`.
- Phase 4 failure counts → existing `outputs/eval/act_spatial_y+5cm/eval_results_w6k2wole.json`, `outputs/residual/y+5cm_sparse/eval_results_o6ukyo53.json`, `outputs/residual/y+5cm_shaped/eval_results_43czuigy.json`.

**Per-seed values gap:** `docs/STATE.md` has mean ± σ per Phase 3 cell but not per-seed values. The hero curve uses ±σ error ribbons sourced from STATE.md. `Cell.per_seed_tsr_custom` is `None` for these cells; only the +5 cm ablation cells have per-seed data populated.

## 5. Filter behavior

| Control | Wiring | Re-render scope | Latency target |
|---|---|---|---|
| Perturbation axis (Both / Spatial / Temporal) | Dash callback | hero curve | <50 ms |
| Metric toggle (TSR_custom / TSR_env / TTS) | Dash callback | hero curve | <50 ms |
| Cell selector | Dash callback | failure-mode panel | <50 ms |
| Failure-mode highlight (legend click) | Plotly native | failure + ablation panels | <16 ms |

All data lives in a `dcc.Store` populated once at boot. Callbacks never re-read from disk.

Callback shape:

```python
@app.callback(
    Output("hero-curve", "figure"),
    Input("axis-filter", "value"),
    Input("metric-toggle", "value"),
    State("data-store", "data"),
)
def update_hero(axis: str, metric: str, data: dict) -> Figure:
    parsed = DashboardData.from_dict(data)
    return build_degradation_curve(parsed, metric=metric, axis_filter=axis)
```

Callbacks are thin wrappers. Figure builders are the unit of test coverage.

## 6. Render budget

| Stage | Budget | Notes |
|---|---|---|
| HF Spaces cold-container wake | excluded | Documented in dashboard README |
| Python + Dash boot | ~600 ms | One-time |
| `load_all()` (4 JSON reads, ~20 KB total) | <50 ms | |
| First HTTP response (HTML shell + JS bundle) | <800 ms | Plotly + DBC bundle ~3 MB gzipped |
| Client-side render of three initial figures | <1200 ms | Pre-rendered server-side as JSON |
| **Visible warm load** | **<2.7 s** | Under PRD's 3 s |

## 7. Edge cases & error handling

- **Empty filter result** — the chosen filter set never produces an empty plot (no need for empty-state design).
- **Malformed `data/headline.json`** — `data.load_headline_json` raises `ValueError`. App startup fails loudly. Pre-deploy CI catches this via `roboeval dashboard build`.
- **Missing `eval_results_*.json` at runtime** — same: loud failure at startup, never at request time.
- **Plotly legend highlight on mobile** — verified by manual test on iPhone Safari; falls back to tap-to-isolate.
- **HF Spaces cold start** — first visit may take ~30 s. Documented; optional uptime ping deferred to follow-up if it materially hurts UX.

## 8. Testing

`tests/dashboard/`:

| File | Tests | Notes |
|---|---|---|
| `test_data.py` | schema-valid headline + ablation + eval_results loaders; malformed inputs raise `ValueError` | covers `roboeval/dashboard/data.py` |
| `test_models.py` | `FailureCounts.total`, `as_fractions` sums to 1, `Cell` rejects unknown axis | invariants |
| `test_figures.py` | axis labels and units present (PRD §9.1 explicit), axis filter selects right subplot count, n_traces matches expected, caption strings contain n_rollouts | mechanical PRD-compliance |
| `test_callbacks.py` | smoke: app imports; each callback round-trips synthetic input → returns `Figure`; `dcc.Store` populated | `dash.testing` gated by `pytest.importorskip("selenium")` |

Gates per [memory: feedback-workflow]:

- `ruff check` — `roboeval/dashboard/**`, `analysis/dashboard/app.py`, `tests/dashboard/**`.
- `ruff format --check` — same.
- `mypy --strict` — `roboeval/dashboard/**` covered by existing `files = ["roboeval"]`; `analysis/dashboard/app.py` added explicitly.
- `pytest -q` — `tests/dashboard/**`.

## 9. Deploy

HF Spaces (Docker SDK):

- `analysis/dashboard/Dockerfile` builds `python:3.11-slim`, installs `requirements.txt` and the local `roboeval` package, exposes port 7860, runs `gunicorn --bind 0.0.0.0:7860 --workers 2 --timeout 60 app:server`.
- HF Space frontmatter in `analysis/dashboard/README.md` (`sdk: docker`, `app_port: 7860`).
- First deploy: manual `hf spaces upload`. CI deploy hook deferred (recorded in §11).
- Top-level `README.md` gets a "Live Demo" badge linking to `huggingface.co/spaces/RubenoDechua/roboeval`.

## 10. CLI surface

```
roboeval dashboard run     # Dash development server on http://localhost:8050
roboeval dashboard build   # validate that all required data files load cleanly;
                           # exits 0 on success, non-zero on schema/file errors.
                           # Used in CI as a pre-deploy data gate.
```

Wired by extending the existing `argparse` subparser tree in `roboeval/cli.py`.

## 11. Out of scope (deferred)

- Per-rollout trajectory replay.
- Multi-policy filter (single-policy v1.0).
- Live W&B fetch (pre-baked data only).
- CI deploy hook to HF Spaces — first deploy is manual; automate after the page is live and stable.
- Uptime ping to prevent HF cold starts — add only if cold-start UX measurably hurts.
- W&B per-seed backfill for Phase 3 cells — error bars from STATE.md ship today; per-seed dots are a v1.1 polish.

## 12. Acceptance

- [ ] Local: `roboeval dashboard run` serves the app on `http://localhost:8050` with all four filters functional.
- [ ] CI: `ruff check`, `ruff format --check`, `mypy --strict`, `pytest -q` all green.
- [ ] HF Space live at `huggingface.co/spaces/RubenoDechua/roboeval`, loads in <3 s warm.
- [ ] All plots have axis labels, units, titles (PRD §9.1 — verified mechanically in `test_figures.py`).
- [ ] Failure-mode legend highlight works on Plotly's native click — no page reload, no Python callback.
- [ ] Mobile (<768 px) layout collapses cleanly; no horizontal scroll.
- [ ] Top-level `README.md` shows a "Live Demo" badge linking to the Space.
- [ ] `docs/STATE.md` updated to reflect Phase 5 dashboard milestone.
