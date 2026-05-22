"""Pure data loaders for the Phase 5 dashboard.

Each loader takes a filesystem path, parses one JSON artifact, and
returns typed dataclasses defined in :mod:`roboeval.dashboard.models`.
No network I/O, no Dash imports — these run cleanly under
``mypy --strict`` and unit-test without a browser.
"""

from __future__ import annotations

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

_HEADLINE_SCHEMA_VERSIONS_SUPPORTED: frozenset[int] = frozenset({1, 2})
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
    """Load ``data/headline.json`` and return the cell tuple.

    Accepts schema v1 and v2; v2 simply adds ``ablation`` and
    ``welch_tests`` siblings that this function ignores (use
    :func:`load_dashboard_data` for the full v2 payload).
    """
    payload = json.loads(Path(path).read_text())
    schema = payload.get("schema_version")
    if schema not in _HEADLINE_SCHEMA_VERSIONS_SUPPORTED:
        raise ValueError(
            f"headline.json schema_version expected one of "
            f"{sorted(_HEADLINE_SCHEMA_VERSIONS_SUPPORTED)}, got {schema!r}"
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


def load_dashboard_data(path: Path) -> DashboardData:
    """Load the full :class:`DashboardData` bundle from ``data/headline.json``.

    Requires schema v2 (cells + ablation + welch_tests inline). This is
    the only loader the runtime dashboard calls; it reads exactly one
    tracked file and has no dependency on gitignored ``outputs/`` or
    ``data/taxonomy/`` artifacts.
    """
    payload = json.loads(Path(path).read_text())
    schema = payload.get("schema_version")
    if schema != 2:
        raise ValueError(
            f"load_dashboard_data requires headline.json schema_version 2, "
            f"got {schema!r}"
        )

    cells = load_headline_json(path)

    ablation: list[AblationCondition] = []
    for raw in payload.get("ablation", ()):
        per_seed = raw["per_seed_means"]
        if len(per_seed) != 3:
            raise ValueError(
                f"ablation condition {raw['condition_id']!r} has "
                f"{len(per_seed)} per_seed_means; expected 3"
            )
        ci = raw["bootstrap_ci"]
        ablation.append(
            AblationCondition(
                condition_id=cast(Literal["A", "B", "C"], raw["condition_id"]),
                label=str(raw["label"]),
                mean_tsr_custom=float(raw["mean_tsr_custom"]),
                std_tsr_custom=float(raw["std_tsr_custom"]),
                per_seed_means=(
                    float(per_seed[0]),
                    float(per_seed[1]),
                    float(per_seed[2]),
                ),
                bootstrap_ci=(float(ci[0]), float(ci[1])),
                failure_counts=_counts_from_dict(raw["failure_counts"]),
                run_id=str(raw["run_id"]),
            )
        )

    welch: list[WelchT] = []
    for raw in payload.get("welch_tests", ()):
        welch.append(
            WelchT(
                arm_id=str(raw["arm_id"]),
                t_statistic=float(raw["t_statistic"]),
                df=float(raw["df"]),
                p_one_sided=float(raw["p_one_sided"]),
            )
        )

    return DashboardData(
        cells=cells,
        ablation=tuple(ablation),
        welch_tests=tuple(welch),
        schema_version=schema,
        generated_at=str(payload.get("generated_at", "")),
    )


def load_all(*, repo_root: Path) -> DashboardData:
    """Aggregate all dashboard data into one :class:`DashboardData`.

    Reads exactly one tracked file (``data/headline.json``), produced
    by ``scripts/build_headline_json.py``. No runtime dependency on the
    gitignored auto_labels or eval_results files.
    """
    return load_dashboard_data(repo_root / "data" / "headline.json")
