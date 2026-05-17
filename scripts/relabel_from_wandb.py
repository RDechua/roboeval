"""Post-hoc auto-label a completed W&B eval run.

Pulls the ``rollouts`` table from a W&B run, reconstructs
:class:`RolloutResult` per row, runs :func:`classify_rollout`, and writes
the same ``data/taxonomy/auto_labels_<run_id>.json`` artifact the in-line
classifier in ``roboeval evaluate`` produces.

PRD §7.3 alignment
------------------
The blinded κ relabel protocol (target κ > 0.6, Landis & Koch
"substantial agreement") requires a frozen auto-labels artifact ≥7 days
before the manual relabel. When classifier rules change after the
evaluation has run — or when the evaluation was run before the in-line
classifier landed — this script regenerates the artifact from the
canonical W&B record without burning GPU hours.

Usage
-----
::

    python scripts/relabel_from_wandb.py <entity/project/run_id>
    python scripts/relabel_from_wandb.py rdechua-uni/roboeval/cm6uf89g

Requires ``wandb login`` to have been run once on the host (standard
``wandb.Api()`` auth).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

from roboeval.evaluation.types import RolloutResult
from roboeval.taxonomy import (
    classify_rollout,
    compute_distribution,
    write_auto_labels,
)


def rollouts_from_wandb_table(table_obj: dict[str, Any]) -> list[RolloutResult]:
    """Reconstruct :class:`RolloutResult` objects from a wandb Table JSON dict.

    The wandb Table on-disk schema is ``{"columns": [...], "data": [[...], ...]}``.
    Column lookup is by name so the function is robust to column-order
    changes in the logger; missing columns default sensibly (zeroes,
    empty strings, or ``None`` for ``terminal_eef_xy_distance_m``).

    Args:
        table_obj: Parsed JSON of a wandb ``rollouts.table.json`` file.

    Returns:
        One :class:`RolloutResult` per row, in table order.
    """
    columns: list[str] = list(table_obj["columns"])
    rows: list[list[Any]] = list(table_obj["data"])
    idx = {col: i for i, col in enumerate(columns)}

    def _get(row: list[Any], key: str, default: Any = None) -> Any:
        if key not in idx:
            return default
        v = row[idx[key]]
        return default if v is None else v

    out: list[RolloutResult] = []
    for row in rows:
        success_step_raw = _get(row, "success_step")
        eef_distance_raw = _get(row, "terminal_eef_xy_distance_m")
        out.append(
            RolloutResult(
                seed_group=int(_get(row, "seed_group", 0)),
                rollout_idx=int(_get(row, "rollout_idx", 0)),
                episode_seed=int(_get(row, "episode_seed", 0)),
                success=bool(_get(row, "success", False)),
                success_custom=bool(_get(row, "success_custom", False)),
                success_step=(
                    None if success_step_raw is None else int(success_step_raw)
                ),
                n_steps=int(_get(row, "n_steps", 0)),
                max_reward=int(_get(row, "max_reward", 0)),
                terminated=bool(_get(row, "terminated", False)),
                truncated=bool(_get(row, "truncated", False)),
                wall_time_s=float(_get(row, "wall_time_s", 0.0)),
                final_cube_z=float(_get(row, "final_cube_z", 0.0)),
                final_cube_x=float(_get(row, "final_cube_x", 0.0)),
                final_cube_y=float(_get(row, "final_cube_y", 0.0)),
                final_cube_xy_dist=float(_get(row, "final_cube_xy_dist", 0.0)),
                failure_mode=str(_get(row, "failure_mode", "")),
                action_sign_flip_rate=float(_get(row, "action_sign_flip_rate", 0.0)),
                terminal_eef_xy_distance_m=(
                    None if eef_distance_raw is None else float(eef_distance_raw)
                ),
                contact_made=bool(_get(row, "contact_made", False)),
                last_50_step_cube_displacement_m=float(
                    _get(row, "last_50_step_cube_displacement_m", 0.0)
                ),
            )
        )
    return out


def _download_rollouts_table(run: Any, dest: Path) -> Path:
    """Fetch the rollouts table JSON file from a W&B run's file list."""
    matches: list[Any] = []
    for f in run.files():
        name = str(f.name)
        if name.endswith(".table.json") and "rollouts" in name:
            matches.append(f)
    if not matches:
        raise FileNotFoundError(
            f"No rollouts.table.json found in run {run.id} — "
            "the run may have logged with a stale schema or no rollouts table."
        )
    # Prefer the most recently-stepped table (wandb appends step suffixes).
    matches.sort(key=lambda f: str(f.name))
    chosen = matches[-1]
    chosen.download(root=str(dest), replace=True)
    return dest / str(chosen.name)


def relabel_run(run_path: str, output_dir: str) -> Path:
    """Pull a W&B run, classify its rollouts, write the auto-labels JSON.

    Args:
        run_path: ``entity/project/run_id`` accepted by ``wandb.Api().run()``.
        output_dir: Directory to write the labels file into.

    Returns:
        Path to the written JSON file.
    """
    import wandb

    api = wandb.Api()
    run: Any = api.run(run_path)  # type: ignore[no-untyped-call]
    config = dict(run.config)

    perturbation_kind = str(config.get("perturbation_kind", "none"))
    perturbation_params_raw = config.get("perturbation_params", {})
    perturbation_params = (
        dict(perturbation_params_raw)
        if isinstance(perturbation_params_raw, dict)
        else {}
    )
    perturbation_applied = perturbation_kind != "none"
    policy_id = str(config.get("policy_id", "unknown"))
    env_id = str(config.get("env_id", "unknown"))

    with tempfile.TemporaryDirectory() as tmpdir:
        table_path = _download_rollouts_table(run, Path(tmpdir))
        table_obj = cast(dict[str, Any], json.loads(table_path.read_text()))

    rollouts = rollouts_from_wandb_table(table_obj)
    labels = [
        classify_rollout(r, perturbation_applied=perturbation_applied) for r in rollouts
    ]
    out_path = write_auto_labels(
        labels,
        output_dir=output_dir,
        run_id=str(run.id),
        config_path=f"<wandb:{run_path}>",
        policy_id=policy_id,
        env_id=env_id,
        perturbation_kind=perturbation_kind,
        perturbation_params=perturbation_params,
        perturbation_applied=perturbation_applied,
    )
    print(f"Wrote {out_path}")
    print(f"  n_rollouts   = {len(labels)}")
    print(f"  failure_dist = {compute_distribution(labels)}")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "run_path",
        help="W&B run identifier: entity/project/run_id "
        "(e.g. rdechua-uni/roboeval/cm6uf89g)",
    )
    parser.add_argument(
        "--out",
        default="data/taxonomy",
        help="Output directory for the auto_labels JSON (default: data/taxonomy).",
    )
    args = parser.parse_args(argv)
    try:
        relabel_run(args.run_path, args.out)
    except Exception as exc:  # noqa: BLE001 - script-level boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
