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
from typing import Literal, cast

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
                axis=cast(
                    Literal["spatial", "temporal", "nominal"],
                    axis,
                ),
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
                    None if raw.get("median_tts") is None else float(raw["median_tts"])
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
                condition_id=cast(Literal["A", "B", "C"], raw["condition_id"]),
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
        # outputs/{eval,residual}/<cell>/eval_results_*.json — repo root is 3 levels up.
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
