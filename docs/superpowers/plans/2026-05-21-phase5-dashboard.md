# Phase 5 Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a hosted, mobile-responsive Plotly/Dash narrative single-page web app that tells the v1.0 RoboEval story (Phase 3 degradation curves + Phase 4 ablation) and meets PRD §9.1 acceptance.

**Architecture:** Split-package. Pure logic (data loaders, dataclasses, figure builders) lives in `roboeval/dashboard/` under `mypy --strict`. The Dash app skeleton (layout + callbacks + server) lives in `analysis/dashboard/` per PRD §5.3. Single `dcc.Store` holds all data; callbacks never re-read from disk.

**Tech Stack:** Python 3.11, Dash 2.17+, Plotly 5.20+, dash-bootstrap-components 1.5+, gunicorn for HF Spaces.

**Workflow gates per commit (from project conventions):** `ruff check`, `ruff format --check`, `mypy --strict roboeval`, `pytest -q`. Commit author `Rubeno Dechua <rubenodechua123@gmail.com>`, no Claude trailers.

---

## File Structure

| Path | Purpose |
|---|---|
| `roboeval/dashboard/__init__.py` | Package init, re-export public dataclasses |
| `roboeval/dashboard/models.py` | Frozen dataclasses: `FailureCounts`, `Cell`, `AblationCondition`, `WelchT`, `DashboardData` |
| `roboeval/dashboard/data.py` | Pure loaders: `load_headline_json`, `load_phase4_ablation`, `load_phase4_eval_results`, `load_all` |
| `roboeval/dashboard/figures.py` | Pure figure builders: `build_degradation_curve`, `build_failure_stack`, `build_phase4_ablation` |
| `analysis/dashboard/app.py` | Dash layout, dcc.Store, callbacks, `server` entry |
| `analysis/dashboard/Dockerfile` | HF Spaces deploy artifact |
| `analysis/dashboard/requirements.txt` | Pinned deploy deps |
| `analysis/dashboard/README.md` | HF Space frontmatter + first-visit note |
| `data/headline.json` | Hand-built tracked artifact with all 11 cells (10 perturbed + nominal) |
| `scripts/build_headline_json.py` | One-time script that builds `data/headline.json` from auto_labels + STATE.md |
| `roboeval/cli.py` | Add `dashboard run` and `dashboard build` subcommands |
| `tests/dashboard/__init__.py` | Test package |
| `tests/dashboard/test_models.py` | Frozen-dataclass invariants |
| `tests/dashboard/test_data.py` | Loader correctness + malformed-input errors |
| `tests/dashboard/test_figures.py` | PRD §9.1 acceptance: axis labels, titles, units, filter behavior |
| `tests/dashboard/test_callbacks.py` | Smoke: app imports, callback round-trips |
| `tests/test_cli.py` | Add tests for the new dashboard subcommands |
| `.github/workflows/ci.yml` | Add `plotly dash dash-bootstrap-components` to install step |
| `README.md` | "Live Demo" badge linking to HF Space |
| `docs/STATE.md` | Phase 5 milestone update |

---

## Task 1: Scaffold dashboard package and test directory

**Files:**
- Create: `roboeval/dashboard/__init__.py`
- Create: `tests/dashboard/__init__.py`

- [ ] **Step 1: Create empty package init**

```python
# roboeval/dashboard/__init__.py
"""Phase 5 interactive dashboard — pure data + figure logic.

The Dash app skeleton lives at ``analysis/dashboard/app.py``;
this package contains only the data loaders, dataclasses, and
Plotly figure builders so they can be unit-tested under
``mypy --strict`` without importing :mod:`dash`.
"""

from __future__ import annotations
```

- [ ] **Step 2: Create empty test package init**

```python
# tests/dashboard/__init__.py
```

- [ ] **Step 3: Verify gates pass on the empty scaffold**

```bash
ruff check roboeval/dashboard tests/dashboard
ruff format --check roboeval/dashboard tests/dashboard
mypy --strict roboeval
pytest -q tests/dashboard
```

Expected: all green. `pytest` reports `no tests ran`.

- [ ] **Step 4: Commit**

```bash
git add roboeval/dashboard/__init__.py tests/dashboard/__init__.py
git commit -m "feat(dashboard): scaffold roboeval.dashboard package"
```

---

## Task 2: Dataclass models with frozen-invariant tests

**Files:**
- Create: `roboeval/dashboard/models.py`
- Create: `tests/dashboard/test_models.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/dashboard/test_models.py
"""Frozen-dataclass invariants for the dashboard data model."""

from __future__ import annotations

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
    with pytest.raises(Exception):  # FrozenInstanceError
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
```

- [ ] **Step 2: Run the test to confirm it fails**

```bash
pytest tests/dashboard/test_models.py -v
```

Expected: ImportError on `roboeval.dashboard.models`.

- [ ] **Step 3: Write the models module**

```python
# roboeval/dashboard/models.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dashboard/test_models.py -v
```

Expected: all 7 tests pass.

- [ ] **Step 5: Run all gates**

```bash
ruff check roboeval/dashboard tests/dashboard
ruff format --check roboeval/dashboard tests/dashboard
mypy --strict roboeval
pytest -q tests/dashboard
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add roboeval/dashboard/models.py tests/dashboard/test_models.py
git commit -m "feat(dashboard): frozen dataclasses for cells, ablation, Welch's t"
```

---

## Task 3: Build script and committed `data/headline.json`

**Files:**
- Create: `scripts/build_headline_json.py`
- Create: `tests/scripts/test_build_headline_json.py`
- Create: `data/headline.json` (committed artifact)

**Context:** This script reads the existing `data/taxonomy/auto_labels_<run_id>.json` files plus the Phase 4 ablation JSON and emits a frozen `data/headline.json`. The mean_tsr / std values for Phase 3 cells come from `docs/STATE.md` (hard-coded in the script for traceability); failure counts come from the auto_labels files. Result: `data/headline.json` is the single source of truth at runtime — the dashboard never re-reads auto_labels.

The 10 Phase 3 cells + nominal cell are mapped to run IDs by reading `config_path` out of each auto_labels file (e.g., `configs/perturbation/spatial/act_spatial_y+5cm.yaml` → cell `y+5cm`).

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_build_headline_json.py
"""Tests for scripts.build_headline_json — the headline.json producer."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_headline_json import build_headline_payload


def test_build_headline_payload_has_eleven_cells(tmp_path: Path) -> None:
    """Smoke test: with the real auto_labels and ablation JSON on disk,
    the builder produces 11 cells (10 perturbed + nominal)."""
    repo_root = Path(__file__).resolve().parents[2]
    payload = build_headline_payload(repo_root=repo_root)
    assert payload["schema_version"] == 1
    cells = payload["cells"]
    assert len(cells) == 11
    axes = {c["axis"] for c in cells}
    assert axes == {"spatial", "temporal", "nominal"}


def test_build_headline_payload_failure_counts_sum_to_n_rollouts(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = build_headline_payload(repo_root=repo_root)
    for cell in payload["cells"]:
        counts = cell["failure_counts"]
        assert sum(counts.values()) == cell["n_rollouts"]


def test_build_headline_payload_spatial_cells_have_known_means(
    tmp_path: Path,
) -> None:
    """Cross-check a few cells against the STATE.md headline table."""
    repo_root = Path(__file__).resolve().parents[2]
    payload = build_headline_payload(repo_root=repo_root)
    by_id = {c["cell_id"]: c for c in payload["cells"]}
    assert by_id["y-5cm"]["mean_tsr_custom"] == 0.127
    assert by_id["y+5cm"]["mean_tsr_custom"] == 0.307
    assert by_id["delay-5step"]["mean_tsr_custom"] == 0.687
    assert by_id["nominal"]["mean_tsr_custom"] == 0.800


def test_headline_json_file_committed_and_valid() -> None:
    """The repository must contain a tracked data/headline.json
    matching the schema produced by the build script."""
    repo_root = Path(__file__).resolve().parents[2]
    headline = json.loads((repo_root / "data" / "headline.json").read_text())
    assert headline["schema_version"] == 1
    assert len(headline["cells"]) == 11
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/scripts/test_build_headline_json.py -v
```

Expected: ImportError on `scripts.build_headline_json`.

- [ ] **Step 3: Write the build script**

```python
# scripts/build_headline_json.py
"""Build the tracked ``data/headline.json`` artifact for the Phase 5 dashboard.

Reads ``data/taxonomy/auto_labels_<run_id>.json`` files (gitignored,
regeneratable from W&B) plus ``docs/figures/phase4_ablation.json`` to
produce ``data/headline.json``, a single committed artifact that the
dashboard loads at runtime.

Per-cell ``mean_tsr_custom`` and ``std_tsr_custom`` for the 10 Phase 3
cells come from ``docs/STATE.md`` and are hard-coded here for traceability
(STATE.md is the human-readable source of truth; this script mechanises
the transcription).

Usage::

    python -m scripts.build_headline_json
    # writes data/headline.json
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("scripts.build_headline_json")

# (mean_tsr_custom, std_tsr_custom) per Phase 3 cell, transcribed from
# docs/STATE.md "Spatial degradation curve" and "Temporal degradation".
_PHASE3_STATS: dict[str, tuple[float, float]] = {
    # spatial
    "y-5cm": (0.127, 0.009),
    "y-3cm": (0.553, 0.025),
    "y-1cm": (0.827, 0.034),
    "y+1cm": (0.720, 0.102),
    "y+3cm": (0.553, 0.041),
    "y+5cm": (0.307, 0.019),
    # nominal anchor
    "nominal": (0.800, 0.057),
    # temporal
    "delay-1step": (0.753, 0.050),
    "delay-3step": (0.767, 0.068),
    "delay-5step": (0.687, 0.066),
}

# Map config_path basenames to canonical cell_ids.
_CONFIG_TO_CELL: dict[str, str] = {
    "act_spatial_y-5cm.yaml": "y-5cm",
    "act_spatial_y-3cm.yaml": "y-3cm",
    "act_spatial_y-1cm.yaml": "y-1cm",
    "act_spatial_y+1cm.yaml": "y+1cm",
    "act_spatial_y+3cm.yaml": "y+3cm",
    "act_spatial_y+5cm.yaml": "y+5cm",
    "act_nominal.yaml": "nominal",
    "act_temporal_delay_1steps.yaml": "delay-1step",
    "act_temporal_delay_3steps.yaml": "delay-3step",
    "act_temporal_delay_5steps.yaml": "delay-5step",
}

# magnitude in cells' natural units (cm for spatial, steps for temporal,
# 0.0 for nominal).
_MAGNITUDES: dict[str, float] = {
    "y-5cm": -0.05,
    "y-3cm": -0.03,
    "y-1cm": -0.01,
    "y+1cm": 0.01,
    "y+3cm": 0.03,
    "y+5cm": 0.05,
    "nominal": 0.0,
    "delay-1step": 1.0,
    "delay-3step": 3.0,
    "delay-5step": 5.0,
}


def _axis_for_cell(cell_id: str) -> str:
    if cell_id == "nominal":
        return "nominal"
    if cell_id.startswith("delay-"):
        return "temporal"
    return "spatial"


def _scan_auto_labels(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Return {cell_id: auto_labels_payload} by config_path lookup."""
    by_cell: dict[str, dict[str, Any]] = {}
    labels_dir = repo_root / "data" / "taxonomy"
    for path in sorted(labels_dir.glob("auto_labels_*.json")):
        payload = json.loads(path.read_text())
        config_basename = Path(payload["config_path"]).name
        cell_id = _CONFIG_TO_CELL.get(config_basename)
        if cell_id is None:
            # Not a Phase 3 cell (probably a Phase 4 residual run).
            continue
        if cell_id in by_cell:
            _LOG.warning(
                "duplicate auto_labels for cell %s: keeping %s, ignoring %s",
                cell_id,
                by_cell[cell_id]["run_id"],
                payload["run_id"],
            )
            continue
        by_cell[cell_id] = payload
    return by_cell


def build_headline_payload(*, repo_root: Path) -> dict[str, Any]:
    """Produce the headline.json payload as a Python dict.

    Args:
        repo_root: Absolute path to the RoboEval repository root.

    Returns:
        The headline.json payload (schema_version 1) ready to be
        ``json.dumps``-ed.

    Raises:
        FileNotFoundError: when expected auto_labels or ablation files
            are missing from disk.
    """
    auto_labels = _scan_auto_labels(repo_root)
    missing = sorted(set(_PHASE3_STATS) - set(auto_labels))
    if missing:
        raise FileNotFoundError(
            f"missing auto_labels for cells {missing!r}; "
            f"regenerate via scripts/relabel_from_wandb.py"
        )

    cells: list[dict[str, Any]] = []
    for cell_id, (mean, std) in _PHASE3_STATS.items():
        payload = auto_labels[cell_id]
        distribution = payload["distribution"]
        n_rollouts = sum(distribution.values())
        cells.append(
            {
                "cell_id": cell_id,
                "axis": _axis_for_cell(cell_id),
                "magnitude": _MAGNITUDES[cell_id],
                "mean_tsr_custom": mean,
                "std_tsr_custom": std,
                "per_seed_tsr_custom": None,
                "mean_tsr": None,
                "median_tts": None,
                "failure_counts": distribution,
                "n_rollouts": n_rollouts,
                "run_id": payload["run_id"],
            }
        )

    return {
        "schema_version": 1,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "source": (
            "Built by scripts/build_headline_json.py from "
            "data/taxonomy/auto_labels_*.json + docs/STATE.md."
        ),
        "cells": cells,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_headline_payload(repo_root=repo_root)
    out_path = repo_root / "data" / "headline.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    _LOG.info("wrote %s (%d cells)", out_path, len(payload["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the build script once locally and commit `data/headline.json`**

```bash
python -m scripts.build_headline_json
```

Expected stdout: `INFO: wrote .../data/headline.json (11 cells)`.

Inspect the file:

```bash
python -c "import json; d=json.load(open('data/headline.json')); print(len(d['cells']), 'cells'); print([c['cell_id'] for c in d['cells']])"
```

Expected: `11 cells` and a list of all 10 perturbed cells + nominal.

- [ ] **Step 5: Confirm `data/headline.json` is NOT gitignored**

```bash
git check-ignore -v data/headline.json
```

Expected: exit code 1 (not ignored). The existing `.gitignore` rule is `data/taxonomy/auto_labels_*.json` — specific enough that `data/headline.json` is tracked.

If exit code 0 (file IS ignored), edit `.gitignore` to narrow the rule and re-confirm.

- [ ] **Step 6: Run the script tests**

```bash
pytest tests/scripts/test_build_headline_json.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 7: Run all gates**

```bash
ruff check scripts tests/scripts
ruff format --check scripts tests/scripts
mypy --strict roboeval   # script is not in mypy scope; verify package still clean
pytest -q
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_headline_json.py tests/scripts/test_build_headline_json.py data/headline.json
git commit -m "feat(dashboard): build_headline_json + tracked data/headline.json

Single committed artifact aggregating Phase 3 cell stats from STATE.md
with failure counts read from the gitignored auto_labels files. The
dashboard reads only data/headline.json at runtime, keeping reproducible
deploys without depending on the per-run W&B artifacts."
```

---

## Task 4: Data loaders

**Files:**
- Create: `roboeval/dashboard/data.py`
- Create: `tests/dashboard/test_data.py`

- [ ] **Step 1: Write the failing test file**

```python
# tests/dashboard/test_data.py
"""Tests for the dashboard data loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboeval.dashboard.data import (
    load_all,
    load_headline_json,
    load_phase4_ablation,
    load_phase4_eval_results,
)
from roboeval.dashboard.models import DashboardData


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_headline_json_returns_eleven_cells() -> None:
    cells = load_headline_json(_repo_root() / "data" / "headline.json")
    assert len(cells) == 11


def test_load_headline_json_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99, "cells": []}))
    with pytest.raises(ValueError, match="schema_version"):
        load_headline_json(bad)


def test_load_phase4_ablation_three_conditions() -> None:
    conds, welches = load_phase4_ablation(
        _repo_root() / "docs" / "figures" / "phase4_ablation.json"
    )
    assert {c.condition_id for c in conds} == {"A", "B", "C"}
    assert {w.arm_id for w in welches} == {"B", "C"}


def test_load_phase4_eval_results_populates_failure_counts() -> None:
    counts_by_condition = load_phase4_eval_results(
        a_path=_repo_root() / "outputs" / "eval" / "act_spatial_y+5cm"
        / "eval_results_w6k2wole.json",
        b_path=_repo_root() / "outputs" / "residual" / "y+5cm_sparse"
        / "eval_results_o6ukyo53.json",
        c_path=_repo_root() / "outputs" / "residual" / "y+5cm_shaped"
        / "eval_results_43czuigy.json",
    )
    assert set(counts_by_condition.keys()) == {"A", "B", "C"}
    assert counts_by_condition["B"].recovery_failure == 106


def test_load_all_returns_dashboard_data() -> None:
    data = load_all(repo_root=_repo_root())
    assert isinstance(data, DashboardData)
    assert len(data.cells) == 11
    assert len(data.ablation) == 3
    assert len(data.welch_tests) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dashboard/test_data.py -v
```

Expected: ImportError on `roboeval.dashboard.data`.

- [ ] **Step 3: Write the data loaders**

```python
# roboeval/dashboard/data.py
"""Pure data loaders for the Phase 5 dashboard.

Each loader takes a filesystem path, parses one JSON artifact, and
returns typed dataclasses defined in :mod:`roboeval.dashboard.models`.
No network I/O, no Dash imports — these run cleanly under
``mypy --strict`` and unit-test without a browser.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, cast

from roboeval.dashboard.models import (
    AblationCondition,
    Cell,
    DashboardData,
    FailureCounts,
    WelchT,
)

_HEADLINE_SCHEMA_VERSION = 1
_ABLATION_SCHEMA_VERSION = 2


def _counts_from_dict(distribution: dict[str, int]) -> FailureCounts:
    return FailureCounts(
        success=int(distribution.get("success", 0)),
        grasp_failure=int(distribution.get("grasp_failure", 0)),
        approach_failure=int(distribution.get("approach_failure", 0)),
        recovery_failure=int(distribution.get("recovery_failure", 0)),
        action_oscillation=int(distribution.get("action_oscillation", 0)),
        timeout=int(distribution.get("timeout", 0)),
        visual_confusion=int(distribution.get("visual_confusion", 0)),
        needs_review=int(distribution.get("needs_review", 0)),
    )


def load_headline_json(path: Path) -> tuple[Cell, ...]:
    """Load ``data/headline.json`` and return the cell tuple."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != _HEADLINE_SCHEMA_VERSION:
        raise ValueError(
            f"headline.json schema_version expected "
            f"{_HEADLINE_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )
    cells: list[Cell] = []
    for raw in payload["cells"]:
        axis = raw["axis"]
        if axis not in ("spatial", "temporal", "nominal"):
            raise ValueError(f"unknown axis {axis!r} for cell {raw.get('cell_id')!r}")
        per_seed = raw.get("per_seed_tsr_custom")
        cells.append(
            Cell(
                cell_id=str(raw["cell_id"]),
                axis=cast(Any, axis),
                magnitude=float(raw["magnitude"]),
                mean_tsr_custom=float(raw["mean_tsr_custom"]),
                std_tsr_custom=float(raw["std_tsr_custom"]),
                per_seed_tsr_custom=(
                    None if per_seed is None else tuple(float(v) for v in per_seed)
                ),
                mean_tsr=(
                    None if raw.get("mean_tsr") is None else float(raw["mean_tsr"])
                ),
                median_tts=(
                    None
                    if raw.get("median_tts") is None
                    else float(raw["median_tts"])
                ),
                failure_counts=_counts_from_dict(raw["failure_counts"]),
                n_rollouts=int(raw["n_rollouts"]),
                run_id=str(raw["run_id"]),
            )
        )
    return tuple(cells)


def load_phase4_ablation(
    path: Path,
) -> tuple[tuple[AblationCondition, ...], tuple[WelchT, ...]]:
    """Load ``docs/figures/phase4_ablation.json`` into typed dataclasses."""
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != _ABLATION_SCHEMA_VERSION:
        raise ValueError(
            f"phase4_ablation.json schema_version expected "
            f"{_ABLATION_SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
        )

    conditions: list[AblationCondition] = []
    for raw in payload["conditions"]:
        per_seed = raw["per_seed_means"]
        if len(per_seed) != 3:
            raise ValueError(
                f"condition {raw['condition_id']!r} has "
                f"{len(per_seed)} per_seed_means; expected 3"
            )
        conditions.append(
            AblationCondition(
                condition_id=cast(Any, raw["condition_id"]),
                label=str(raw["label"]),
                mean_tsr_custom=float(raw["mean"]),
                std_tsr_custom=float(raw["std"]),
                per_seed_means=(
                    float(per_seed[0]),
                    float(per_seed[1]),
                    float(per_seed[2]),
                ),
                bootstrap_ci=(
                    float(raw["bootstrap_ci_low"]),
                    float(raw["bootstrap_ci_high"]),
                ),
                # Filled in later by load_all() from eval_results JSONs.
                failure_counts=FailureCounts(0, 0, 0, 0, 0, 0, 0, 0),
                run_id=str(raw["run_ids"][0]),
            )
        )

    welches: list[WelchT] = []
    for raw in payload["comparisons"]:
        welches.append(
            WelchT(
                arm_id=str(raw["condition_id"]),
                t_statistic=float(raw["t_statistic"]),
                df=float(raw["df"]),
                p_one_sided=float(raw["p_value"]),
            )
        )

    return tuple(conditions), tuple(welches)


def load_phase4_eval_results(
    *, a_path: Path, b_path: Path, c_path: Path
) -> dict[str, FailureCounts]:
    """Read the three Phase 4 eval_results JSONs.

    The eval_results JSONs do not embed the failure-mode distribution
    directly — the distribution lives in the corresponding
    ``data/taxonomy/auto_labels_<run_id>.json``. This loader walks each
    eval_results -> auto_labels by sibling lookup of the run_id.
    """
    by_cond: dict[str, FailureCounts] = {}
    for cond_id, path in (("A", a_path), ("B", b_path), ("C", c_path)):
        eval_payload = json.loads(Path(path).read_text())
        run_id = eval_payload["run_id"]
        repo_root = Path(path).resolve().parents[3]
        labels_path = repo_root / "data" / "taxonomy" / f"auto_labels_{run_id}.json"
        labels_payload = json.loads(labels_path.read_text())
        by_cond[cond_id] = _counts_from_dict(labels_payload["distribution"])
    return by_cond


def load_all(*, repo_root: Path) -> DashboardData:
    """Aggregate all dashboard data sources into one :class:`DashboardData`."""
    cells = load_headline_json(repo_root / "data" / "headline.json")
    ablation, welches = load_phase4_ablation(
        repo_root / "docs" / "figures" / "phase4_ablation.json"
    )
    counts = load_phase4_eval_results(
        a_path=repo_root
        / "outputs"
        / "eval"
        / "act_spatial_y+5cm"
        / "eval_results_w6k2wole.json",
        b_path=repo_root
        / "outputs"
        / "residual"
        / "y+5cm_sparse"
        / "eval_results_o6ukyo53.json",
        c_path=repo_root
        / "outputs"
        / "residual"
        / "y+5cm_shaped"
        / "eval_results_43czuigy.json",
    )
    ablation_with_counts = tuple(
        AblationCondition(
            condition_id=c.condition_id,
            label=c.label,
            mean_tsr_custom=c.mean_tsr_custom,
            std_tsr_custom=c.std_tsr_custom,
            per_seed_means=c.per_seed_means,
            bootstrap_ci=c.bootstrap_ci,
            failure_counts=counts[c.condition_id],
            run_id=c.run_id,
        )
        for c in ablation
    )
    return DashboardData(
        cells=cells,
        ablation=ablation_with_counts,
        welch_tests=welches,
        schema_version=_HEADLINE_SCHEMA_VERSION,
        generated_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    )
```

- [ ] **Step 2 (rerun): Verify tests pass**

```bash
pytest tests/dashboard/test_data.py -v
```

Expected: all 5 tests pass.

- [ ] **Step 3: Run all gates**

```bash
ruff check roboeval/dashboard tests/dashboard
ruff format --check roboeval/dashboard tests/dashboard
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 4: Commit**

```bash
git add roboeval/dashboard/data.py tests/dashboard/test_data.py
git commit -m "feat(dashboard): JSON loaders with schema-version validation"
```

---

## Task 5: `build_degradation_curve` figure builder

**Files:**
- Create: `roboeval/dashboard/figures.py` (new module; this task adds the first function)
- Create: `tests/dashboard/test_figures.py` (new test module; this task adds the first three tests)

- [ ] **Step 1: Write the failing tests**

```python
# tests/dashboard/test_figures.py
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
        Cell("y-5cm", "spatial", -0.05, 0.127, 0.009, None, None, None,
             FailureCounts(19, 0, 7, 121, 0, 0, 0, 3), 150, "rid1"),
        Cell("y-3cm", "spatial", -0.03, 0.553, 0.025, None, None, None,
             FailureCounts(83, 0, 0, 63, 0, 0, 0, 4), 150, "rid2"),
        Cell("y-1cm", "spatial", -0.01, 0.827, 0.034, None, None, None,
             FailureCounts(124, 0, 0, 26, 0, 0, 0, 0), 150, "rid3"),
        Cell("nominal", "nominal", 0.0, 0.800, 0.057, None, None, None,
             FailureCounts(120, 0, 0, 0, 0, 28, 0, 2), 150, "rid_nominal"),
        Cell("y+1cm", "spatial", 0.01, 0.720, 0.102, None, None, None,
             FailureCounts(108, 0, 0, 37, 0, 0, 0, 5), 150, "rid4"),
        Cell("y+3cm", "spatial", 0.03, 0.553, 0.041, None, None, None,
             FailureCounts(83, 0, 1, 56, 0, 0, 0, 10), 150, "rid5"),
        Cell("y+5cm", "spatial", 0.05, 0.307, 0.019, None, None, None,
             FailureCounts(46, 0, 1, 89, 0, 0, 0, 13), 150, "rid6"),
        Cell("delay-1step", "temporal", 1.0, 0.753, 0.050, None, None, None,
             FailureCounts(113, 0, 0, 33, 0, 0, 0, 4), 150, "rid7"),
        Cell("delay-3step", "temporal", 3.0, 0.767, 0.068, None, None, None,
             FailureCounts(115, 0, 0, 32, 0, 0, 0, 3), 150, "rid8"),
        Cell("delay-5step", "temporal", 5.0, 0.687, 0.066, None, None, None,
             FailureCounts(103, 0, 0, 45, 0, 0, 0, 2), 150, "rid9"),
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
    # Two subplots: spatial (xaxis) and temporal (xaxis2).
    assert layout.xaxis.title.text is not None
    assert layout.xaxis2.title.text is not None
    assert layout.yaxis.title.text is not None
    # Units must appear in xaxis titles per PRD §9.1.
    assert "cm" in layout.xaxis.title.text.lower()
    assert "step" in layout.xaxis2.title.text.lower()


def test_degradation_curve_axis_filter_spatial_hides_temporal() -> None:
    fig = build_degradation_curve(
        _make_data(), metric="mean_tsr_custom", axis_filter="spatial"
    )
    # Mean line + error ribbon = 2 traces per axis when both shown;
    # spatial-only should have exactly 2 traces (mean + ribbon) for one panel
    # plus zero traces for the temporal subplot.
    visible_traces = [tr for tr in fig.data if tr.visible is not False]
    # Spatial panel: mean + error band = 2 traces.
    assert len(visible_traces) == 2


def test_degradation_curve_metric_toggle_changes_y_title() -> None:
    fig_custom = build_degradation_curve(
        _make_data(), metric="mean_tsr_custom", axis_filter="both"
    )
    fig_env = build_degradation_curve(
        _make_data(), metric="mean_tsr", axis_filter="both"
    )
    assert fig_custom.layout.yaxis.title.text != fig_env.layout.yaxis.title.text
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/dashboard/test_figures.py -v
```

Expected: ImportError on `roboeval.dashboard.figures`.

- [ ] **Step 3: Write the figures module + degradation curve builder**

```python
# roboeval/dashboard/figures.py
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

from roboeval.dashboard.models import (
    AblationCondition,
    DashboardData,
    WelchT,
)

Metric = Literal["mean_tsr_custom", "mean_tsr", "median_tts"]
AxisFilter = Literal["both", "spatial", "temporal"]

_METRIC_LABELS: dict[Metric, str] = {
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
    xs = [c.magnitude * 100.0 for c in cells]  # convert m → cm for display
    ys = [getattr(c, "mean_tsr_custom") for c in cells]  # only metric in v1.0
    sigmas = [c.std_tsr_custom for c in cells]
    if metric == "mean_tsr":
        ys = [c.mean_tsr if c.mean_tsr is not None else c.mean_tsr_custom for c in cells]
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
        ys = [c.mean_tsr if c.mean_tsr is not None else c.mean_tsr_custom for c in cells]
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
            name=f"{name} ±σ",
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
    """Hero figure: TSR vs perturbation magnitude, side-by-side spatial + temporal."""
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
    fig.update_xaxes(
        title_text="Action delay (steps)", row=1, col=2, zeroline=True
    )
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/dashboard/test_figures.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Gates**

```bash
ruff check roboeval/dashboard tests/dashboard
ruff format --check roboeval/dashboard tests/dashboard
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add roboeval/dashboard/figures.py tests/dashboard/test_figures.py
git commit -m "feat(dashboard): hero degradation curve builder + axis-label tests"
```

---

## Task 6: `build_failure_stack` figure builder

**Files:**
- Modify: `roboeval/dashboard/figures.py` (add function)
- Modify: `tests/dashboard/test_figures.py` (add tests)

- [ ] **Step 1: Add the failing tests at the end of `tests/dashboard/test_figures.py`**

```python
# Append to tests/dashboard/test_figures.py

from roboeval.dashboard.figures import build_failure_stack  # noqa: E402


def test_failure_stack_has_six_visible_categories_for_recovery_dominant_cell() -> None:
    data = _make_data()
    fig = build_failure_stack(data, cell_id="y+5cm")
    # 6 PRD-named failure categories with non-zero or near-zero presence;
    # we expect at least Success + Recovery + Approach + NeedsReview present.
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
```

- [ ] **Step 2: Run to verify fails**

```bash
pytest tests/dashboard/test_figures.py -v
```

Expected: ImportError on `build_failure_stack`.

- [ ] **Step 3: Add the builder to `roboeval/dashboard/figures.py`**

Append after `build_degradation_curve`:

```python
# Append to roboeval/dashboard/figures.py

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
        fig.add_trace(
            go.Bar(
                x=[cell.cell_id],
                y=[value_pct],
                name=display,
                marker_color=color,
                hovertemplate=(
                    f"{display}: %{{y:.1f}}% ({int(value_pct * cell.n_rollouts / 100)} rollouts)"
                    "<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        barmode="stack",
        title=f"Failure-mode breakdown — {cell.cell_id} ({cell.n_rollouts} rollouts, 3 seeds)",
        template="plotly_white",
        yaxis_title="Fraction of rollouts (%)",
        xaxis_title="Cell",
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
        height=380,
        legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "x": 1.02},
    )
    fig.update_yaxes(range=[0, 100])
    return fig
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/dashboard/test_figures.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Gates**

```bash
ruff check roboeval/dashboard tests/dashboard
ruff format --check roboeval/dashboard tests/dashboard
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add roboeval/dashboard/figures.py tests/dashboard/test_figures.py
git commit -m "feat(dashboard): per-cell failure-mode stacked-bar builder"
```

---

## Task 7: `build_phase4_ablation` figure builder

**Files:**
- Modify: `roboeval/dashboard/figures.py`
- Modify: `tests/dashboard/test_figures.py`

- [ ] **Step 1: Add the failing tests**

```python
# Append to tests/dashboard/test_figures.py

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
    # 8 stacked traces (one per category), each carrying 3 x values (A, B, C).
    for tr in fig.data:
        assert list(tr.x) == ["A", "B", "C"]


def test_phase4_ablation_title_mentions_plus_5cm() -> None:
    fig = build_phase4_ablation(_make_ablation_data())
    assert fig.layout.title is not None
    assert "+5" in fig.layout.title.text or "+5cm" in fig.layout.title.text
```

- [ ] **Step 2: Run to confirm fails**

```bash
pytest tests/dashboard/test_figures.py -v
```

Expected: ImportError on `build_phase4_ablation`.

- [ ] **Step 3: Append the builder to `roboeval/dashboard/figures.py`**

```python
# Append to roboeval/dashboard/figures.py

def build_phase4_ablation(data: DashboardData) -> go.Figure:
    """Stacked bar comparing failure-mode distribution across A/B/C at +5 cm."""
    if not data.ablation:
        raise ValueError("DashboardData has no ablation conditions")
    fig = go.Figure()
    x = [c.condition_id for c in data.ablation]
    for key, display, color in _FAILURE_MODE_ORDER:
        ys = [
            c.failure_counts.as_fractions()[key] * 100.0 for c in data.ablation
        ]
        fig.add_trace(
            go.Bar(
                x=x,
                y=ys,
                name=display,
                marker_color=color,
            )
        )
    fig.update_layout(
        barmode="stack",
        title="Phase 4 ablation — failure-mode distribution at +5cm spatial",
        template="plotly_white",
        yaxis_title="Fraction of rollouts (%)",
        xaxis_title="Condition (A=base, B=sparse, C=shaped)",
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
        height=380,
        legend={"orientation": "v", "yanchor": "middle", "y": 0.5, "x": 1.02},
    )
    fig.update_yaxes(range=[0, 100])
    return fig
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/dashboard/test_figures.py -v
```

Expected: 8 tests pass.

- [ ] **Step 5: Gates**

```bash
ruff check roboeval/dashboard tests/dashboard
ruff format --check roboeval/dashboard tests/dashboard
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add roboeval/dashboard/figures.py tests/dashboard/test_figures.py
git commit -m "feat(dashboard): Phase 4 ablation 3-condition stacked-bar builder"
```

---

## Task 8: Dash app skeleton (layout + dcc.Store, no callbacks yet)

**Files:**
- Create: `analysis/dashboard/app.py`
- Create: `tests/dashboard/test_callbacks.py` (with one passing smoke test)

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/dashboard/test_callbacks.py
"""Smoke tests for the Dash app skeleton at analysis/dashboard/app.py."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

dash = pytest.importorskip("dash")

_APP_PATH = Path(__file__).resolve().parents[2] / "analysis" / "dashboard"


def _load_app_module():  # type: ignore[no-untyped-def]
    if str(_APP_PATH) not in sys.path:
        sys.path.insert(0, str(_APP_PATH))
    return importlib.import_module("app")


def test_app_module_imports_and_exposes_server() -> None:
    mod = _load_app_module()
    assert hasattr(mod, "server")
    assert hasattr(mod, "app")


def test_app_layout_has_hero_curve_and_failure_stack() -> None:
    mod = _load_app_module()
    layout_html = str(mod.app.layout)
    assert "hero-curve" in layout_html
    assert "failure-stack" in layout_html
    assert "ablation-plot" in layout_html
```

- [ ] **Step 2: Run to confirm fails**

```bash
pytest tests/dashboard/test_callbacks.py -v
```

Expected: ImportError on `app` module.

- [ ] **Step 3: Write the Dash app skeleton**

```python
# analysis/dashboard/app.py
"""RoboEval Phase 5 interactive dashboard (Plotly/Dash, HF Spaces).

Loads ``DashboardData`` once at boot, keeps it in a client-side ``dcc.Store``,
and renders a narrative single-page layout: hero degradation curves,
per-cell failure-mode breakdown, Phase 4 ablation panel.

Run locally::

    roboeval dashboard run

Deploy: see ``analysis/dashboard/README.md``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html

from roboeval.dashboard.data import load_all
from roboeval.dashboard.figures import (
    build_degradation_curve,
    build_failure_stack,
    build_phase4_ablation,
)
from roboeval.dashboard.models import DashboardData


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _data_to_store_dict(data: DashboardData) -> dict[str, Any]:
    """Serialise DashboardData to a JSON-safe dict for ``dcc.Store``."""

    def _convert(value: Any) -> Any:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {f.name: _convert(getattr(value, f.name)) for f in dataclasses.fields(value)}
        if isinstance(value, (list, tuple)):
            return [_convert(v) for v in value]
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        return value

    return _convert(data)  # type: ignore[no-any-return]


def _store_dict_to_data(payload: dict[str, Any]) -> DashboardData:
    """Reverse of ``_data_to_store_dict`` — rebuild dataclasses from a dict."""
    raw_json = json.dumps(payload)
    # Round-trip through our loader by writing a tmp file would be heavy;
    # instead reconstruct directly.
    from roboeval.dashboard.models import (
        AblationCondition,
        Cell,
        FailureCounts,
        WelchT,
    )

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
                None if c["per_seed_tsr_custom"] is None
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


def _build_layout() -> html.Div:
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
                    html.A("GitHub", href="https://github.com/RubenoDechua/roboeval",
                           target="_blank", className="me-3"),
                    html.A("PRD", href="https://github.com/RubenoDechua/roboeval/blob/main/docs/PRD.md",
                           target="_blank"),
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
                        xs=12, md=6,
                    ),
                    dbc.Col(
                        dbc.RadioItems(
                            id="metric-toggle",
                            options=[
                                {"label": "TSR (custom)", "value": "mean_tsr_custom"},
                                {"label": "TSR (env)", "value": "mean_tsr"},
                                {"label": "TTS", "value": "median_tts"},
                            ],
                            value="mean_tsr_custom",
                            inline=True,
                        ),
                        xs=12, md=6,
                    ),
                ],
                className="mb-2",
            ),
            dcc.Graph(id="hero-curve",
                      figure=build_degradation_curve(
                          _DATA, metric="mean_tsr_custom", axis_filter="both"
                      )),

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
                    xs=12, md=4,
                ),
                className="mb-2",
            ),
            dcc.Graph(id="failure-stack",
                      figure=build_failure_stack(_DATA, cell_id="y+5cm")),

            html.Hr(),
            html.H2("Phase 4 ablation at +5 cm spatial"),
            dcc.Graph(id="ablation-plot", figure=build_phase4_ablation(_DATA)),

            html.Hr(),
            html.Details(
                [
                    html.Summary("Methods & reproducibility"),
                    html.Ul([
                        html.Li(f"3 seeds × 50 rollouts per cell; n={c.n_rollouts}")
                        for c in _DATA.cells[:3]
                    ]),
                    html.P(f"Generated at {_DATA.generated_at}"),
                ],
                className="mt-3",
            ),
        ],
    )


app = Dash(__name__, external_stylesheets=[dbc.themes.LITERA])
app.title = "RoboEval — Phase 5 dashboard"
app.layout = _build_layout()
server = app.server
```

- [ ] **Step 4: Verify smoke tests pass**

```bash
pytest tests/dashboard/test_callbacks.py -v
```

Expected: 2 tests pass.

- [ ] **Step 5: Gates**

```bash
ruff check analysis/dashboard tests/dashboard
ruff format --check analysis/dashboard tests/dashboard
mypy --strict roboeval   # analysis/dashboard not in strict scope yet
pytest -q
```

Expected: all green. (We'll bring `analysis/dashboard/` into mypy --strict in Task 11.)

- [ ] **Step 6: Local run sanity check**

```bash
python analysis/dashboard/app.py 2>&1 | tee /tmp/dashboard.log &
sleep 4
curl -sI http://localhost:8050/ | head -1
kill %1 2>/dev/null
```

Expected: `HTTP/1.0 200 OK`. If the page 500s, read `/tmp/dashboard.log` and fix before commit.

- [ ] **Step 7: Commit**

```bash
git add analysis/dashboard/app.py tests/dashboard/test_callbacks.py
git commit -m "feat(dashboard): Dash app skeleton with dcc.Store + three charts"
```

---

## Task 9: Wire interactive callbacks

**Files:**
- Modify: `analysis/dashboard/app.py`
- Modify: `tests/dashboard/test_callbacks.py`

- [ ] **Step 1: Add the failing tests**

```python
# Append to tests/dashboard/test_callbacks.py

def test_update_hero_callback_round_trip() -> None:
    mod = _load_app_module()
    fig_dict = mod.update_hero(
        axis="spatial",
        metric="mean_tsr_custom",
        data=mod._STORE_PAYLOAD,
    )
    # Plotly figures returned from callbacks are dicts.
    assert "data" in fig_dict
    assert "layout" in fig_dict


def test_update_failure_stack_callback_changes_with_cell() -> None:
    mod = _load_app_module()
    a = mod.update_failure_stack(cell_id="y+5cm", data=mod._STORE_PAYLOAD)
    b = mod.update_failure_stack(cell_id="y-5cm", data=mod._STORE_PAYLOAD)
    assert a["layout"]["title"]["text"] != b["layout"]["title"]["text"]
```

- [ ] **Step 2: Run to confirm fails**

```bash
pytest tests/dashboard/test_callbacks.py -v
```

Expected: `AttributeError: module 'app' has no attribute 'update_hero'`.

- [ ] **Step 3: Append callbacks to `analysis/dashboard/app.py`**

```python
# Append to analysis/dashboard/app.py, after `app.layout = _build_layout()`:


@app.callback(
    Output("hero-curve", "figure"),
    Input("axis-filter", "value"),
    Input("metric-toggle", "value"),
    State("data-store", "data"),
)
def update_hero(axis: str, metric: str, data: dict[str, Any]) -> dict[str, Any]:
    parsed = _store_dict_to_data(data)
    fig = build_degradation_curve(
        parsed, metric=metric, axis_filter=axis  # type: ignore[arg-type]
    )
    return fig.to_dict()


@app.callback(
    Output("failure-stack", "figure"),
    Input("cell-select", "value"),
    State("data-store", "data"),
)
def update_failure_stack(cell_id: str, data: dict[str, Any]) -> dict[str, Any]:
    parsed = _store_dict_to_data(data)
    fig = build_failure_stack(parsed, cell_id=cell_id)
    return fig.to_dict()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
```

- [ ] **Step 4: Verify callbacks tests pass**

```bash
pytest tests/dashboard/test_callbacks.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Gates**

```bash
ruff check analysis/dashboard tests/dashboard
ruff format --check analysis/dashboard tests/dashboard
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 6: Visual smoke test in the browser**

```bash
python analysis/dashboard/app.py &
sleep 4
echo "Open http://localhost:8050 — toggle axis filter, metric, cell. Confirm:"
echo "  - axis filter hides one subplot"
echo "  - metric toggle updates y-axis label"
echo "  - cell selector updates the failure-mode panel title"
echo "When done, kill the server below."
read -r _
kill %1
```

If anything misbehaves, fix before committing.

- [ ] **Step 7: Commit**

```bash
git add analysis/dashboard/app.py tests/dashboard/test_callbacks.py
git commit -m "feat(dashboard): wire axis / metric / cell-selector callbacks"
```

---

## Task 10: CLI subcommands `roboeval dashboard run` and `roboeval dashboard build`

**Files:**
- Modify: `roboeval/cli.py`
- Modify: `tests/test_cli.py`

- [ ] **Step 1: Add failing CLI tests**

```python
# Append to tests/test_cli.py

def test_dashboard_build_subcommand_zero_exit_on_valid_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from roboeval.cli import main

    exit_code = main(["dashboard", "build"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "11" in out  # cells loaded


def test_dashboard_run_subcommand_dry_run_zero_exit() -> None:
    """`dashboard run --dry-run` should exit 0 without starting the server."""
    from roboeval.cli import main

    exit_code = main(["dashboard", "run", "--dry-run"])
    assert exit_code == 0
```

- [ ] **Step 2: Run to confirm fails**

```bash
pytest tests/test_cli.py -v -k dashboard
```

Expected: argparse rejects `dashboard` as an unknown subcommand.

- [ ] **Step 3: Extend `roboeval/cli.py`**

Find the `subparsers.add_parser("residual", ...)` block in `_build_parser`. After it, add:

```python
# Insert into roboeval/cli.py inside _build_parser(), after the residual block:

    dashboard = subparsers.add_parser(
        "dashboard",
        help=(
            "Phase 5 interactive dashboard: build static data artifact, "
            "or launch the Dash dev server."
        ),
    )
    dashboard_sub = dashboard.add_subparsers(dest="dashboard_cmd", required=True)

    dashboard_sub.add_parser(
        "build",
        help=(
            "Validate that all dashboard data sources load cleanly. "
            "Exits 0 on success, non-zero on schema or file errors."
        ),
    )

    dashboard_run = dashboard_sub.add_parser(
        "run",
        help="Launch the Dash development server on http://localhost:8050.",
    )
    dashboard_run.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1).",
    )
    dashboard_run.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to bind (default: 8050).",
    )
    dashboard_run.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the app constructs without starting the server.",
    )
```

Then add command dispatchers anywhere above `main()`:

```python
# Insert into roboeval/cli.py above main():

def _cmd_dashboard_build() -> int:
    from pathlib import Path

    from roboeval.dashboard.data import load_all

    repo_root = Path(__file__).resolve().parents[1]
    data = load_all(repo_root=repo_root)
    print(
        f"[roboeval dashboard build] OK — "
        f"{len(data.cells)} cells, "
        f"{len(data.ablation)} ablation conditions, "
        f"{len(data.welch_tests)} Welch's t tests."
    )
    return 0


def _cmd_dashboard_run(*, host: str, port: int, dry_run: bool) -> int:
    import importlib
    import sys
    from pathlib import Path

    app_dir = Path(__file__).resolve().parents[1] / "analysis" / "dashboard"
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))
    mod = importlib.import_module("app")
    if dry_run:
        print("[roboeval dashboard run] OK — app constructs cleanly (dry-run).")
        return 0
    mod.app.run(host=host, port=port, debug=False)
    return 0
```

Then wire them into `main()` — add after the `residual` branch:

```python
# Modify main() in roboeval/cli.py — add this branch after the residual one:

    if args.command == "dashboard":
        if args.dashboard_cmd == "build":
            return _cmd_dashboard_build()
        if args.dashboard_cmd == "run":
            return _cmd_dashboard_run(
                host=str(args.host),
                port=int(args.port),
                dry_run=bool(args.dry_run),
            )
        raise AssertionError(f"unhandled dashboard_cmd: {args.dashboard_cmd!r}")
```

- [ ] **Step 4: Verify tests pass**

```bash
pytest tests/test_cli.py -v -k dashboard
```

Expected: 2 tests pass.

- [ ] **Step 5: Gates**

```bash
ruff check roboeval tests
ruff format --check roboeval tests
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add roboeval/cli.py tests/test_cli.py
git commit -m "feat(cli): roboeval dashboard {run,build} subcommands"
```

---

## Task 11: HF Spaces deploy artifacts

**Files:**
- Create: `analysis/dashboard/Dockerfile`
- Create: `analysis/dashboard/requirements.txt`
- Create: `analysis/dashboard/README.md`

- [ ] **Step 1: Write `analysis/dashboard/requirements.txt`**

```
# analysis/dashboard/requirements.txt
plotly>=5.20
dash>=2.17
dash-bootstrap-components>=1.5
gunicorn>=21.2
```

The local `roboeval` package is installed via the Dockerfile's `pip install -e .` step (no need to re-list its transitive deps; only the dashboard-specific runtime deps).

- [ ] **Step 2: Write `analysis/dashboard/Dockerfile`**

```dockerfile
# analysis/dashboard/Dockerfile
# Hugging Face Spaces Docker SDK target.
#
# Build context is REPO ROOT, not analysis/dashboard/. HF Spaces clones
# the repo and runs `docker build` with the dockerfile pointed at, so
# the COPY paths are repo-relative.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY roboeval /app/roboeval
COPY analysis/dashboard /app/analysis/dashboard
COPY data/headline.json /app/data/headline.json
COPY docs/figures/phase4_ablation.json /app/docs/figures/phase4_ablation.json
COPY outputs/eval/act_spatial_y+5cm/eval_results_w6k2wole.json \
     /app/outputs/eval/act_spatial_y+5cm/eval_results_w6k2wole.json
COPY outputs/residual/y+5cm_sparse/eval_results_o6ukyo53.json \
     /app/outputs/residual/y+5cm_sparse/eval_results_o6ukyo53.json
COPY outputs/residual/y+5cm_shaped/eval_results_43czuigy.json \
     /app/outputs/residual/y+5cm_shaped/eval_results_43czuigy.json

# The data loader looks up auto_labels by sibling lookup; copy the
# three Phase 4 auto_labels in so load_phase4_eval_results works.
COPY data/taxonomy/auto_labels_w6k2wole.json /app/data/taxonomy/auto_labels_w6k2wole.json
COPY data/taxonomy/auto_labels_o6ukyo53.json /app/data/taxonomy/auto_labels_o6ukyo53.json
COPY data/taxonomy/auto_labels_43czuigy.json /app/data/taxonomy/auto_labels_43czuigy.json

RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -r analysis/dashboard/requirements.txt

ENV PYTHONPATH=/app

EXPOSE 7860
CMD ["gunicorn", "--chdir", "analysis/dashboard", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--timeout", "60", \
     "app:server"]
```

- [ ] **Step 3: Write `analysis/dashboard/README.md`** (HF Space frontmatter)

```markdown
---
title: RoboEval
emoji: 🤖
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# RoboEval — Phase 5 Dashboard

Interactive dashboard for the RoboEval failure-mode and residual-RL study
of ACT on AlohaTransferCube. See the main repo at
[github.com/RubenoDechua/roboeval](https://github.com/RubenoDechua/roboeval)
for the full PRD, research log, and source.

**First visit:** if the Space has been idle for >48h, the container takes
about 30 seconds to wake. Subsequent loads complete in <3 seconds.
```

- [ ] **Step 4: Verify the Dockerfile builds locally** (optional — only if Docker is installed)

```bash
docker build -t roboeval-dashboard -f analysis/dashboard/Dockerfile . && \
docker run --rm -p 7860:7860 roboeval-dashboard &
sleep 8
curl -sI http://localhost:7860/ | head -1
docker stop $(docker ps -q --filter ancestor=roboeval-dashboard) 2>/dev/null
```

Expected: `HTTP/1.0 200 OK`. If Docker is not installed locally, skip this step — the smoke test runs on HF Spaces in Task 12.

- [ ] **Step 5: Gates**

```bash
ruff check analysis tests
ruff format --check analysis tests
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add analysis/dashboard/Dockerfile analysis/dashboard/requirements.txt analysis/dashboard/README.md
git commit -m "feat(dashboard): HF Spaces Docker artifacts (Dockerfile, reqs, README)"
```

---

## Task 12: CI, top-level README badge, STATE.md update, manual deploy

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `README.md`
- Modify: `docs/STATE.md`

- [ ] **Step 1: Add dashboard deps to CI install step**

Open `.github/workflows/ci.yml` and modify the `Install dev tools and minimal runtime deps` step:

```yaml
# .github/workflows/ci.yml — modify the existing pip install line:

      - name: Install dev tools and minimal runtime deps
        run: |
          python -m pip install --upgrade pip
          pip install --index-url https://download.pytorch.org/whl/cpu \
                      "torch>=2.0"
          pip install "ruff>=0.5,<0.6" "mypy>=1.10" "pytest>=8.0" \
                      "numpy>=1.26" "gymnasium>=0.29" "omegaconf>=2.3" \
                      "pyyaml>=6.0" "wandb>=0.16" \
                      "plotly>=5.20" "dash>=2.17" \
                      "dash-bootstrap-components>=1.5"
```

- [ ] **Step 2: Add "Live Demo" badge to top-level `README.md`**

Open `README.md` and immediately under the `# RoboEval` title insert:

```markdown
[![Live Demo](https://img.shields.io/badge/HF%20Spaces-Live%20Demo-blue?logo=huggingface)](https://huggingface.co/spaces/RubenoDechua/roboeval)
```

(If the Space URL ends up at a different namespace, update the link in this step.)

- [ ] **Step 3: Update `docs/STATE.md`** — add a Phase 5 section under Phase, refresh the Next-session intent

Open `docs/STATE.md` and append after the existing "Phase 4 closed" line in the `## Phase` section:

```markdown
**Phase 5 (Communication) in progress**: interactive Plotly/Dash dashboard
landed (`analysis/dashboard/` + `roboeval/dashboard/`), deployed to HF
Spaces. Demo video and arXiv-style writeup remain.
```

Replace the existing `## Next session intent` content with:

```markdown
## Next session intent

Phase 5 dashboard live. Remaining Phase 5 work:

1. **90-second demo video** — narrated, side-by-side nominal vs +5 cm
   Recovery rollout with residual overlay (PRD §9.1 demo deliverable).
2. **Blog / arXiv-style writeup** — builds on `docs/phase4_ablation.md`
   and the Phase 3 cross-axis findings; honest-null framing.
3. **MkDocs site** — static-site wrapper around PRD, research-log,
   phase4_ablation.md, plus auto-generated API docs.
4. **κ relabel** when 2026-05-24 unlocks (samples already exported).

v1.1 design backlog unchanged from Phase 4 close.
```

- [ ] **Step 4: Run all gates**

```bash
ruff check .
ruff format --check .
mypy --strict roboeval
pytest -q
```

Expected: all green.

- [ ] **Step 5: Commit and push**

```bash
git add .github/workflows/ci.yml README.md docs/STATE.md
git commit -m "feat(dashboard): CI deps, README live-demo badge, STATE.md Phase 5 entry"
git push origin main
```

- [ ] **Step 6: Manual HF Spaces deploy**

```bash
# 1. Create the Space (one-time) via the HF web UI at
#    https://huggingface.co/new-space — SDK: Docker, name: roboeval.
# 2. Clone the Space repo locally:
git clone https://huggingface.co/spaces/RubenoDechua/roboeval /tmp/hf-roboeval
# 3. Mirror dashboard artifacts + tracked data into the Space repo:
cd /tmp/hf-roboeval
rsync -a --delete \
  --exclude '.git' \
  --include 'analysis/' --include 'analysis/dashboard/' --include 'analysis/dashboard/**' \
  --include 'roboeval/' --include 'roboeval/**' \
  --include 'pyproject.toml' --include 'README.md' \
  --include 'data/' --include 'data/headline.json' \
  --include 'data/taxonomy/' \
  --include 'data/taxonomy/auto_labels_w6k2wole.json' \
  --include 'data/taxonomy/auto_labels_o6ukyo53.json' \
  --include 'data/taxonomy/auto_labels_43czuigy.json' \
  --include 'docs/' --include 'docs/figures/' \
  --include 'docs/figures/phase4_ablation.json' \
  --include 'outputs/' --include 'outputs/eval/' --include 'outputs/eval/act_spatial_y+5cm/' \
  --include 'outputs/eval/act_spatial_y+5cm/eval_results_w6k2wole.json' \
  --include 'outputs/residual/' \
  --include 'outputs/residual/y+5cm_sparse/' --include 'outputs/residual/y+5cm_sparse/eval_results_o6ukyo53.json' \
  --include 'outputs/residual/y+5cm_shaped/' --include 'outputs/residual/y+5cm_shaped/eval_results_43czuigy.json' \
  --exclude '*' \
  /Users/rubenodehcua/Desktop/roboeval/ ./
# 4. Move the HF Space README into place (analysis/dashboard/README.md becomes the Space's README)
cp analysis/dashboard/README.md README.md
git add -A
git commit -m "Initial deploy"
git push
# 5. Visit https://huggingface.co/spaces/RubenoDechua/roboeval — wait for build, verify <3s warm load.
```

- [ ] **Step 7: Verify acceptance criteria**

Tick each of these against the live Space (per spec §12):

- [ ] Local: `roboeval dashboard run` serves on `:8050` with all four filters functional.
- [ ] CI green on push.
- [ ] HF Space loads in <3 s warm.
- [ ] All plots have axis labels, units, titles (tested mechanically).
- [ ] Failure-mode legend highlight works on Plotly's native click — no Python callback.
- [ ] Mobile (<768 px) layout collapses cleanly; no horizontal scroll (manual test on iPhone Safari).
- [ ] Top-level `README.md` shows a "Live Demo" badge linking to the Space.
- [ ] `docs/STATE.md` updated to reflect Phase 5 milestone.

If any item fails, file a follow-up issue and fix in a follow-on commit rather than amending.

---

## Self-Review (filled in inline after writing the plan)

**Spec coverage:**

| Spec section | Plan tasks |
|---|---|
| §1 Goal | Tasks 1–12 collectively |
| §2 Story arc | Task 8 (layout) |
| §3 Architecture | Tasks 1, 2, 4, 5, 6, 7, 8 |
| §4 Data model | Task 2 |
| §5 Filter behavior | Tasks 8, 9 |
| §6 Render budget | Task 9 (smoke test step), Task 12 (acceptance) |
| §7 Edge cases | Task 4 (schema_version), Task 6 (unknown cell) |
| §8 Testing | Tasks 2, 4, 5, 6, 7, 8, 9, 10 (interspersed throughout) |
| §9 Deploy | Tasks 11, 12 |
| §10 CLI surface | Task 10 |
| §11 Out of scope | (deferred; no tasks) |
| §12 Acceptance | Task 12 step 7 (checklist against live Space) |

**Placeholder scan:** none — every step has concrete code or commands.

**Type consistency:** `Cell`/`AblationCondition`/`FailureCounts` names match across Tasks 2, 4, 5, 6, 7, 8, 9. `axis_filter` / `metric` parameter names match between Task 5 figure builder and Task 9 callback. `cell_id` is the consistent identifier across Task 6 (figure builder) and Task 9 (callback).

**Risks noted but not blocking:**
- HF Spaces cold-start UX is documented in `analysis/dashboard/README.md` rather than engineered away. If recruiters bounce due to cold start, add an uptime ping (out of scope here).
- The Dockerfile bundles outputs/* JSONs that are already tracked in git. If the user prefers a smaller image, swap `COPY` of full eval_results for a pre-extracted minimal subset — defer to v1.1.
- The first deploy script in Task 12 step 6 is interactive (one-time setup). It is left as a runnable shell snippet rather than an automated CI job; CI deploy is deferred per spec §11.
