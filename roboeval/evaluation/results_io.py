"""Persisted-results schema and writer for eval runs.

Mirrors :mod:`roboeval.taxonomy.io`: a pure ``*_to_json_obj`` builder
(round-trippable, no I/O, easy to unit-test) plus a thin ``write_*``
wrapper that creates the parent directory and serialises with
``json.dumps(..., indent=2)``.

Why a separate persisted artifact when W&B already stores everything?
Three reasons:

1. **Offline-resilience.** ``configs/baseline/act_nominal_fast.yaml``
   runs with ``wandb.mode=disabled``; CI is offline; some user runs may
   fail at W&B sync time. The eval result is the deliverable and must
   land on disk regardless of W&B reachability.
2. **Ablation aggregation** (PRD section 8.3) needs to read all 3
   conditions * 3 seeds * 50 rollouts back into one process to compute
   delta-TSR and Welch's t-test. Pulling from the W&B API would couple the
   aggregator to network state and credentials; reading local JSON does
   not.
3. **Reproducible re-analysis.** Failure-mode reanalysis, plot
   regeneration, and the eventual write-up all want per-rollout fields
   (cube xy, contact_made, displacement, …) without re-running the env.
   The schema below dumps the full :class:`~roboeval.evaluation.types.EvalResult`
   so downstream scripts never need to re-rollout.

Schema is versioned (``schema_version: 1``); any field rename or
semantic change requires bumping the version and updating the
aggregator's reader.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from roboeval.evaluation.types import EvalResult, RolloutResult

SCHEMA_VERSION = 1


def _rollout_to_dict(rollout: RolloutResult) -> dict[str, Any]:
    """Serialise one :class:`~roboeval.evaluation.types.RolloutResult`.

    Booleans are coerced explicitly (gym-aloha sometimes hands back
    ``numpy.bool_`` which is *not* JSON-serialisable by ``json.dumps``).
    """
    return {
        "seed_group": int(rollout.seed_group),
        "rollout_idx": int(rollout.rollout_idx),
        "episode_seed": int(rollout.episode_seed),
        "success": bool(rollout.success),
        "success_custom": bool(rollout.success_custom),
        "success_step": (
            None if rollout.success_step is None else int(rollout.success_step)
        ),
        "n_steps": int(rollout.n_steps),
        "max_reward": int(rollout.max_reward),
        "terminated": bool(rollout.terminated),
        "truncated": bool(rollout.truncated),
        "wall_time_s": float(rollout.wall_time_s),
        "final_cube_x": float(rollout.final_cube_x),
        "final_cube_y": float(rollout.final_cube_y),
        "final_cube_z": float(rollout.final_cube_z),
        "final_cube_xy_dist": float(rollout.final_cube_xy_dist),
        "failure_mode": str(rollout.failure_mode),
        "action_sign_flip_rate": float(rollout.action_sign_flip_rate),
        "terminal_eef_xy_distance_m": (
            None
            if rollout.terminal_eef_xy_distance_m is None
            else float(rollout.terminal_eef_xy_distance_m)
        ),
        "contact_made": bool(rollout.contact_made),
        "last_50_step_cube_displacement_m": float(
            rollout.last_50_step_cube_displacement_m
        ),
    }


def eval_result_to_json_obj(
    result: EvalResult,
    *,
    run_id: str,
    config_path: str,
    git_sha: str,
    timestamp: str,
    policy_kind: str,
    device: str,
    seeds: list[int],
    n_rollouts_per_seed: int,
    max_steps: int,
    perturbation_kind: str,
    perturbation_params: Mapping[str, Any],
    success_criterion: Mapping[str, Any] | None = None,
    residual: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the schema-v1 JSON object for one eval run.

    Pure data — no I/O. Symmetric across baseline (``policy_kind="act"``)
    and residual (``policy_kind="residual_act"``) runs so the ablation
    aggregator can read either with one loader.

    Args:
        result: Aggregated :class:`EvalResult` from
            :func:`roboeval.evaluation.loop.evaluate_policy`.
        run_id: Stable identifier (W&B run id when available, otherwise
            the timestamped ``run_name``). Used as the filename suffix
            in :func:`write_eval_results`.
        config_path: Path to the YAML config that drove the run.
        git_sha: HEAD short SHA at run time (use ``"unknown"`` outside
            a git working tree).
        timestamp: ISO-8601 timestamp of the run.
        policy_kind: ``"act"`` for Condition A, ``"residual_act"`` for
            Conditions B/C, etc.
        device: Inference device (``"mps"`` / ``"cpu"`` / …).
        seeds: Seed groups passed to ``evaluate_policy``.
        n_rollouts_per_seed: ``cfg.eval.n_rollouts_per_seed``.
        max_steps: ``cfg.eval.max_steps``.
        perturbation_kind: ``"none"`` or the perturbation kind (
            ``spatial`` / ``visual`` / …).
        perturbation_params: Perturbation kwargs (e.g. ``{"dx_m": 0.0,
            "dy_m": 0.05}``); empty dict when none.
        success_criterion: Optional dict with the geometric criterion
            (``z_threshold_m``, ``xy_tolerance_m``, ``dwell_steps``,
            ``target_xy``). Omitted from the JSON when ``None``.
        residual: Optional dict with residual-specific metadata (``path``,
            ``reward_kind``, ``alpha_init``, ``log_std_init``). Omitted
            when ``None`` (Condition A).

    Returns:
        JSON-serialisable dict matching schema v1.
    """
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_path": config_path,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "policy_kind": policy_kind,
        "policy_id": result.policy_id,
        "env_id": result.env_id,
        "device": device,
        "seeds": [int(s) for s in seeds],
        "n_rollouts_per_seed": int(n_rollouts_per_seed),
        "max_steps": int(max_steps),
        "perturbation_kind": perturbation_kind,
        "perturbation_params": dict(perturbation_params),
        "metrics": {
            "mean_tsr": float(result.mean_tsr),
            "std_tsr": float(result.std_tsr),
            "mean_tsr_custom": float(result.mean_tsr_custom),
            "std_tsr_custom": float(result.std_tsr_custom),
            "median_tts": (
                None if result.median_tts is None else float(result.median_tts)
            ),
            "n_rollouts": int(result.n_rollouts),
            "n_seed_groups": int(result.n_seed_groups),
            "per_seed_tsr": [float(x) for x in result.per_seed_tsr],
            "per_seed_tsr_custom": [float(x) for x in result.per_seed_tsr_custom],
        },
        "rollouts": [_rollout_to_dict(r) for r in result.rollouts],
    }
    if success_criterion is not None:
        payload["success_criterion"] = dict(success_criterion)
    if residual is not None:
        payload["residual"] = dict(residual)
    return payload


def write_eval_results(
    result: EvalResult,
    output_dir: Path | str,
    *,
    run_id: str,
    config_path: str,
    git_sha: str,
    timestamp: str,
    policy_kind: str,
    device: str,
    seeds: list[int],
    n_rollouts_per_seed: int,
    max_steps: int,
    perturbation_kind: str,
    perturbation_params: Mapping[str, Any],
    success_criterion: Mapping[str, Any] | None = None,
    residual: Mapping[str, Any] | None = None,
) -> Path:
    """Write schema-v1 eval results to ``<output_dir>/eval_results_<run_id>.json``.

    Creates the parent directory with ``parents=True`` if missing.
    Distinct ``run_id`` values per call avoid clobbering; the convention
    matches :func:`roboeval.taxonomy.io.write_auto_labels` so an eval
    run produces a paired ``eval_results_<id>.json`` and
    ``auto_labels_<id>.json`` in their respective output dirs.

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_results_{run_id}.json"
    obj = eval_result_to_json_obj(
        result,
        run_id=run_id,
        config_path=config_path,
        git_sha=git_sha,
        timestamp=timestamp,
        policy_kind=policy_kind,
        device=device,
        seeds=seeds,
        n_rollouts_per_seed=n_rollouts_per_seed,
        max_steps=max_steps,
        perturbation_kind=perturbation_kind,
        perturbation_params=perturbation_params,
        success_criterion=success_criterion,
        residual=residual,
    )
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


__all__ = [
    "SCHEMA_VERSION",
    "eval_result_to_json_obj",
    "write_eval_results",
]
