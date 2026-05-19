"""Schema + writer tests for ``eval_results_<run_id>.json``.

Mirrors the structure of ``tests/taxonomy/test_io.py`` so reviewers
familiar with that test see the same pattern (pure builder round-trips,
writer creates parents, distinct ``run_id`` avoids clobbering).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboeval.evaluation.results_io import (
    SCHEMA_VERSION,
    eval_result_to_json_obj,
    write_eval_results,
)
from roboeval.evaluation.types import RolloutResult, aggregate


def _rollout(
    seed_group: int,
    idx: int,
    *,
    success: bool,
    success_custom: bool = False,
    step: int | None = None,
) -> RolloutResult:
    return RolloutResult(
        seed_group=seed_group,
        rollout_idx=idx,
        episode_seed=seed_group * 100_003 + idx,
        success=success,
        success_custom=success_custom,
        success_step=step,
        n_steps=step or 400,
        max_reward=4 if success else 0,
        terminated=success,
        truncated=not success,
        wall_time_s=1.0,
        final_cube_z=0.1 if success else 0.0,
        final_cube_x=0.01,
        final_cube_y=0.02,
        final_cube_xy_dist=0.0224,
        failure_mode="" if success else "recovery_failure",
        action_sign_flip_rate=0.3,
        terminal_eef_xy_distance_m=0.05,
        contact_made=True,
        last_50_step_cube_displacement_m=0.001,
    )


def _make_result(per_seed_success_counts: list[int]) -> object:
    """Build an EvalResult with three seed groups, ``per_seed_success_counts[k]``
    primary successes (and the same number of custom successes) in group ``k``,
    out of 5 rollouts per group."""
    rollouts = []
    for seed_idx, n_success in enumerate(per_seed_success_counts):
        for r_idx in range(5):
            succ = r_idx < n_success
            rollouts.append(
                _rollout(
                    seed_idx,
                    r_idx,
                    success=succ,
                    success_custom=succ,
                    step=20 if succ else None,
                )
            )
    return aggregate(rollouts, policy_id="lerobot/act_test", env_id="aloha/test-v0")


def test_schema_round_trips_baseline_payload(tmp_path: Path) -> None:
    """Condition A payload: no residual block, with success_criterion."""
    result = _make_result([3, 4, 2])
    payload = eval_result_to_json_obj(
        result,
        run_id="abc123",
        config_path="configs/perturbation/spatial/act_spatial_y+5cm.yaml",
        git_sha="deadbee",
        timestamp="2026-05-19T10:00:00+00:00",
        policy_kind="act",
        device="mps",
        seeds=[0, 1, 2],
        n_rollouts_per_seed=5,
        max_steps=500,
        perturbation_kind="spatial",
        perturbation_params={"dx_m": 0.0, "dy_m": 0.05},
        success_criterion={
            "z_threshold_m": 0.05,
            "xy_tolerance_m": 0.022,
            "dwell_steps": 5,
            "target_xy": [0.0, 0.01],
        },
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["run_id"] == "abc123"
    assert payload["policy_kind"] == "act"
    assert "residual" not in payload
    assert payload["success_criterion"]["xy_tolerance_m"] == pytest.approx(0.022)
    assert payload["perturbation_kind"] == "spatial"
    assert payload["perturbation_params"] == {"dx_m": 0.0, "dy_m": 0.05}

    # Aggregate metrics: 3, 4, 2 of 5 → 0.6, 0.8, 0.4 → mean=0.6.
    metrics = payload["metrics"]
    assert metrics["mean_tsr_custom"] == pytest.approx(0.6)
    assert metrics["per_seed_tsr_custom"] == pytest.approx([0.6, 0.8, 0.4])
    assert metrics["n_rollouts"] == 15
    assert metrics["n_seed_groups"] == 3

    # Per-rollout dump preserved in order, fields populated.
    assert len(payload["rollouts"]) == 15
    first = payload["rollouts"][0]
    assert first["seed_group"] == 0
    assert first["episode_seed"] == 0
    assert first["success"] is True
    assert first["failure_mode"] == ""
    last = payload["rollouts"][-1]
    assert last["seed_group"] == 2
    assert last["failure_mode"] == "recovery_failure"
    assert last["success_step"] is None

    # Round-trip through JSON to catch un-serialisable fields.
    rehydrated = json.loads(json.dumps(payload))
    assert rehydrated == payload


def test_schema_residual_block_present(tmp_path: Path) -> None:
    """Condition B/C payload includes the residual metadata block."""
    result = _make_result([4, 5, 3])
    payload = eval_result_to_json_obj(
        result,
        run_id="run_b_seed0",
        config_path="configs/residual/residual_ppo_y+5cm_sparse.yaml",
        git_sha="cafef00",
        timestamp="2026-05-19T11:00:00+00:00",
        policy_kind="residual_act",
        device="mps",
        seeds=[0, 1, 2],
        n_rollouts_per_seed=5,
        max_steps=500,
        perturbation_kind="spatial",
        perturbation_params={"dx_m": 0.0, "dy_m": 0.05},
        residual={
            "path": "outputs/residual/y+5cm_sparse/ppo_residual.zip",
            "reward_kind": "sparse",
            "alpha_init": 0.05,
            "log_std_init": -2.0,
        },
    )
    assert payload["policy_kind"] == "residual_act"
    assert payload["residual"]["reward_kind"] == "sparse"
    assert payload["residual"]["alpha_init"] == pytest.approx(0.05)
    assert payload["residual"]["log_std_init"] == pytest.approx(-2.0)


def test_writer_creates_parent_and_filename(tmp_path: Path) -> None:
    result = _make_result([3, 3, 3])
    nested = tmp_path / "outputs" / "residual" / "y+5cm_sparse"
    assert not nested.exists()
    path = write_eval_results(
        result,
        output_dir=nested,
        run_id="abc123xy",
        config_path="cfg.yaml",
        git_sha="abc1234",
        timestamp="2026-05-19T10:00:00+00:00",
        policy_kind="residual_act",
        device="cpu",
        seeds=[0, 1, 2],
        n_rollouts_per_seed=5,
        max_steps=500,
        perturbation_kind="spatial",
        perturbation_params={"dy_m": 0.05},
    )
    assert path.exists()
    assert path.name == "eval_results_abc123xy.json"
    assert path.parent == nested
    loaded = json.loads(path.read_text())
    assert loaded["schema_version"] == SCHEMA_VERSION
    assert loaded["metrics"]["mean_tsr_custom"] == pytest.approx(0.6)


def test_writer_distinct_run_ids_dont_clobber(tmp_path: Path) -> None:
    result = _make_result([2, 3, 4])
    common_kwargs = {
        "config_path": "cfg.yaml",
        "git_sha": "abc1234",
        "timestamp": "2026-05-19T10:00:00+00:00",
        "policy_kind": "act",
        "device": "cpu",
        "seeds": [0, 1, 2],
        "n_rollouts_per_seed": 5,
        "max_steps": 500,
        "perturbation_kind": "none",
        "perturbation_params": {},
    }
    p1 = write_eval_results(result, tmp_path, run_id="r1", **common_kwargs)
    p2 = write_eval_results(result, tmp_path, run_id="r2", **common_kwargs)
    assert p1 != p2
    assert p1.exists() and p2.exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "eval_results_r1.json",
        "eval_results_r2.json",
    ]
