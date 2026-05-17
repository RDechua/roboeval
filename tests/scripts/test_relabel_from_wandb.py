"""Tests for scripts/relabel_from_wandb.py table-row reconstruction.

The wandb-fetch path requires network credentials and isn't exercised in
CI; what we test here is the pure-function bit — given a wandb Table JSON
dict, do we get back :class:`RolloutResult` objects the classifier can
consume?
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ isn't a package, so add it to sys.path for the import below.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from relabel_from_wandb import rollouts_from_wandb_table  # noqa: E402

_COLUMNS = [
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


def _row(**overrides):
    base = {
        "seed_group": 0,
        "rollout_idx": 0,
        "episode_seed": 0,
        "success": False,
        "success_custom": False,
        "success_step": None,
        "n_steps": 400,
        "max_reward": 0,
        "terminated": False,
        "truncated": True,
        "wall_time_s": 8.0,
        "final_cube_z": 0.0,
        "final_cube_x": 0.0,
        "final_cube_y": 0.0,
        "final_cube_xy_dist": 0.0,
        "failure_mode": "",
        "action_sign_flip_rate": 0.0,
        "terminal_eef_xy_distance_m": None,
        "contact_made": False,
        "last_50_step_cube_displacement_m": 0.0,
    }
    base.update(overrides)
    return [base[col] for col in _COLUMNS]


def test_table_to_rollouts_preserves_fields():
    table = {
        "columns": _COLUMNS,
        "data": [
            _row(
                seed_group=2,
                rollout_idx=7,
                episode_seed=200017,
                success=True,
                success_custom=True,
                success_step=120,
                n_steps=125,
                max_reward=4,
                terminated=True,
                truncated=False,
                wall_time_s=14.3,
                final_cube_z=0.06,
                final_cube_x=-0.018,
                final_cube_y=0.50,
                final_cube_xy_dist=0.50,
                action_sign_flip_rate=0.05,
                terminal_eef_xy_distance_m=0.03,
                contact_made=True,
                last_50_step_cube_displacement_m=0.001,
            ),
        ],
    }
    rollouts = rollouts_from_wandb_table(table)
    assert len(rollouts) == 1
    r = rollouts[0]
    assert r.seed_group == 2
    assert r.rollout_idx == 7
    assert r.episode_seed == 200017
    assert r.success is True
    assert r.success_step == 120
    assert r.max_reward == 4
    assert r.terminated is True
    assert r.truncated is False
    assert r.wall_time_s == pytest.approx(14.3)
    assert r.final_cube_z == pytest.approx(0.06)
    assert r.action_sign_flip_rate == pytest.approx(0.05)
    assert r.terminal_eef_xy_distance_m == pytest.approx(0.03)
    assert r.contact_made is True
    assert r.last_50_step_cube_displacement_m == pytest.approx(0.001)


def test_table_to_rollouts_preserves_null_success_step_and_distance():
    table = {
        "columns": _COLUMNS,
        "data": [
            _row(
                success=False,
                success_step=None,
                terminal_eef_xy_distance_m=None,
            ),
        ],
    }
    rollouts = rollouts_from_wandb_table(table)
    assert rollouts[0].success_step is None
    assert rollouts[0].terminal_eef_xy_distance_m is None


def test_table_to_rollouts_handles_missing_optional_columns():
    # An older wandb table with no trajectory aggregates still parses;
    # missing columns fall back to the dataclass defaults (zeroes / False /
    # None for the distance).
    legacy_cols = [
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
    ]
    table = {
        "columns": legacy_cols,
        "data": [
            [
                0,
                0,
                0,
                True,
                True,
                100,
                100,
                4,
                True,
                False,
                5.0,
                0.06,
                0.0,
                0.5,
                0.5,
                "",
            ]
        ],
    }
    rollouts = rollouts_from_wandb_table(table)
    assert rollouts[0].success is True
    assert rollouts[0].action_sign_flip_rate == 0.0
    assert rollouts[0].terminal_eef_xy_distance_m is None
    assert rollouts[0].contact_made is False
    assert rollouts[0].last_50_step_cube_displacement_m == 0.0


def test_table_to_rollouts_round_trips_through_classifier():
    from roboeval.taxonomy import FailureMode, classify_rollout

    table = {
        "columns": _COLUMNS,
        "data": [
            _row(success=True, success_step=120, terminated=True, truncated=False),
            _row(
                success=False,
                contact_made=True,
                final_cube_z=0.005,
                truncated=True,
                last_50_step_cube_displacement_m=0.0,
            ),
        ],
    }
    rollouts = rollouts_from_wandb_table(table)
    labels = [classify_rollout(r) for r in rollouts]
    assert labels[0].failure_mode is None
    assert labels[1].failure_mode == FailureMode.GRASP_FAILURE
