"""W&B logging wrappers for RoboEval evaluation runs.

Implements the schema locked in the Week 2 plan:

* Run config logged at init via ``wandb.init(config=...)`` AND uploaded as a
  ``wandb.Artifact`` for permanent reproducibility (experiment-logger skill
  requirement: "Config must be logged as a W&B artifact at the start of
  every run").
* Per-rollout rows appended to a ``wandb.Table`` named ``rollouts``.
* Summary scalars (mean/std TSR primary and custom, median TTS, counts)
  written to ``run.summary``.

All wandb objects are typed ``Any`` because wandb does not publish stubs;
the file is exempted from ruff ANN401 in ``pyproject.toml``.
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from roboeval.evaluation.types import EvalResult, RolloutResult

_LOG = logging.getLogger("roboeval.evaluation.logger")

_ROLLOUT_TABLE_COLUMNS: list[str] = [
    "seed_group",
    "rollout_idx",
    "episode_seed",
    "success",
    "success_custom",
    "success_step",
    "n_steps",
    "max_reward",
    "terminated",
    "truncated",
    "wall_time_s",
    "final_cube_z",
    "final_cube_x",
    "final_cube_y",
    "final_cube_xy_dist",
    "failure_mode",
    "action_sign_flip_rate",
    "terminal_eef_xy_distance_m",
    "contact_made",
    "last_50_step_cube_displacement_m",
]


class WandbRunHandle:
    """Active W&B run plus the in-flight rollouts table.

    Returned by the :func:`wandb_run` context manager; do not construct
    directly. Per-rollout rows are buffered locally until
    :meth:`log_summary` flushes the table into the run.
    """

    def __init__(self, run: Any, rollout_table: Any) -> None:
        """Bind a wandb.Run and a wandb.Table together.

        Args:
            run: The ``wandb.Run`` object returned by ``wandb.init``.
            rollout_table: A ``wandb.Table`` constructed with the locked
                column list.
        """
        self._run = run
        self._table = rollout_table

    def log_rollout(self, result: RolloutResult) -> None:
        """Append one row to the rollouts table.

        Args:
            result: One per-rollout outcome to record.
        """
        self._table.add_data(
            result.seed_group,
            result.rollout_idx,
            result.episode_seed,
            result.success,
            result.success_custom,
            result.success_step,
            result.n_steps,
            result.max_reward,
            result.terminated,
            result.truncated,
            result.wall_time_s,
            result.final_cube_z,
            result.final_cube_x,
            result.final_cube_y,
            result.final_cube_xy_dist,
            result.failure_mode,
            result.action_sign_flip_rate,
            result.terminal_eef_xy_distance_m,
            result.contact_made,
            result.last_50_step_cube_displacement_m,
        )

    def log_summary(self, eval_result: EvalResult) -> None:
        """Write aggregate scalars to run.summary and flush the rollouts table.

        Args:
            eval_result: Aggregated metrics for the whole run.
        """
        summary = self._run.summary
        summary["mean_tsr"] = eval_result.mean_tsr
        summary["std_tsr"] = eval_result.std_tsr
        summary["mean_tsr_custom"] = eval_result.mean_tsr_custom
        summary["std_tsr_custom"] = eval_result.std_tsr_custom
        summary["median_tts"] = eval_result.median_tts
        summary["n_rollouts"] = eval_result.n_rollouts
        summary["n_seed_groups"] = eval_result.n_seed_groups
        summary["per_seed_tsr"] = list(eval_result.per_seed_tsr)
        summary["per_seed_tsr_custom"] = list(eval_result.per_seed_tsr_custom)
        self._run.log({"rollouts": self._table})

    @property
    def url(self) -> str | None:
        """Public W&B run URL when online; ``None`` for offline/disabled runs."""
        url = getattr(self._run, "url", None)
        return cast(str | None, url) if url else None

    @property
    def run_id(self) -> str | None:
        """W&B run id; ``None`` if not assigned (e.g. disabled mode)."""
        rid = getattr(self._run, "id", None)
        return cast(str | None, rid) if rid else None

    def log_distribution(self, distribution: Mapping[str, int]) -> None:
        """Write failure-mode counts to run.summary.

        PRD §7.3 step 5 ("per-policy failure distribution heatmap")
        consumes these summary scalars when building the cell-by-cell
        heatmap across runs.

        Args:
            distribution: ``{bucket_name: count}`` from
                :func:`roboeval.taxonomy.io.compute_distribution`.
        """
        summary = self._run.summary
        for bucket, count in distribution.items():
            summary[f"failure_dist/{bucket}"] = count


def _upload_config_artifact(run: Any, config: Mapping[str, Any], name: str) -> None:
    """Save the run config as a JSON artifact for reproducibility.

    Args:
        run: Active wandb.Run.
        config: The same dict passed to ``wandb.init(config=...)``.
        name: Human-readable artifact name; defaults to
            ``"eval_config_<run.id>"`` if empty.
    """
    import wandb

    artifact = wandb.Artifact(name=name, type="eval_config")
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "config.json"
        # Make values JSON-serialisable (cast tuples/None safely).
        path.write_text(json.dumps(dict(config), indent=2, default=str))
        artifact.add_file(str(path), name="config.json")
        run.log_artifact(artifact)


@contextmanager
def wandb_run(
    project: str,
    name: str,
    config: Mapping[str, Any],
    tags: list[str] | None = None,
    mode: str = "online",
) -> Iterator[WandbRunHandle]:
    """Context manager wrapping ``wandb.init`` / ``wandb.finish``.

    Performs three things at entry: starts the run with the given config,
    uploads that config as a W&B artifact for permanent retrieval, and
    prepares an empty rollouts table to receive per-rollout rows.

    Args:
        project: W&B project name.
        name: Run name (timestamped suffix is the convention upstream).
        config: Run config dict — policy id, env id, seeds, success
            thresholds, device, lerobot version, etc.
        tags: Optional list of run tags.
        mode: One of ``"online"``, ``"offline"``, ``"disabled"``. The last
            value is what tests use.

    Yields:
        A :class:`WandbRunHandle` for per-rollout and summary logging.
    """
    import wandb

    valid_modes = {"online", "offline", "disabled", "shared"}
    if mode not in valid_modes:
        raise ValueError(f"mode must be one of {valid_modes}; got {mode!r}")

    run = wandb.init(
        project=project,
        name=name,
        config=dict(config),
        tags=tags,
        mode=mode,  # type: ignore[arg-type]  # validated against valid_modes above
    )
    if run is None:
        raise RuntimeError("wandb.init() returned None — check WANDB_MODE / auth.")

    # Best-effort config artifact upload. Disabled-mode runs no-op; offline
    # runs save to the local wandb/ dir; online runs upload to the cloud.
    try:
        _upload_config_artifact(run, config, name=f"eval_config_{run.id}")
    except Exception as exc:  # noqa: BLE001 - artifact upload is non-critical
        _LOG.warning("Could not upload config artifact: %s", exc)

    table = wandb.Table(columns=_ROLLOUT_TABLE_COLUMNS)
    handle = WandbRunHandle(run, table)
    try:
        yield handle
    finally:
        run.finish()
