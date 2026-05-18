"""Unit tests for the residual RL reward functions (PRD §8.2)."""

from __future__ import annotations

import numpy as np
import pytest

from roboeval.residual.reward import (
    combined_reward,
    shaped_distance_reward,
    sparse_success_reward,
)


def test_sparse_success_reward_returns_one_on_success():
    assert sparse_success_reward({"is_success": True}) == 1.0


def test_sparse_success_reward_returns_zero_on_failure():
    assert sparse_success_reward({"is_success": False}) == 0.0


def test_sparse_success_reward_handles_missing_key():
    assert sparse_success_reward({}) == 0.0


def test_shaped_distance_reward_is_zero_at_target():
    target = (-0.018, 0.506)
    cube_xy = np.array(target, dtype=np.float64)
    assert shaped_distance_reward(cube_xy, target) == pytest.approx(0.0, abs=1e-9)


def test_shaped_distance_reward_is_negative_away_from_target():
    target = (0.0, 0.0)
    cube_xy = np.array([0.03, 0.04], dtype=np.float64)  # distance 0.05
    assert shaped_distance_reward(cube_xy, target) == pytest.approx(-0.05, abs=1e-6)


def test_shaped_distance_reward_uses_l2():
    """A 3-4-5 triangle: cube at (3,4), target at (0,0) -> -5."""
    cube_xy = np.array([3.0, 4.0], dtype=np.float64)
    assert shaped_distance_reward(cube_xy, (0.0, 0.0)) == pytest.approx(-5.0)


def test_combined_reward_sparse_only_when_weight_zero():
    info = {"is_success": True}
    cube = np.array([1.0, 0.0], dtype=np.float64)
    r = combined_reward(info, cube, (0.0, 0.0), shaping_weight=0.0)
    assert r == 1.0  # sparse-only; distance term ignored


def test_combined_reward_adds_shaping_term():
    info = {"is_success": False}
    cube = np.array([3.0, 4.0], dtype=np.float64)  # 5m from origin
    r = combined_reward(info, cube, (0.0, 0.0), shaping_weight=0.1)
    # 0 (sparse) + 0.1 * (-5.0) = -0.5
    assert r == pytest.approx(-0.5)


def test_combined_reward_success_dominates_at_small_shaping_weight():
    info = {"is_success": True}
    cube = np.array([3.0, 4.0], dtype=np.float64)
    r = combined_reward(info, cube, (0.0, 0.0), shaping_weight=0.01)
    # 1.0 + 0.01 * (-5.0) = 0.95
    assert r == pytest.approx(0.95)


def test_combined_reward_at_calibrated_target_gives_just_sparse():
    """Realistic case: cube reaches calibrated target; shaping term is 0."""
    target = (-0.018, 0.506)
    info = {"is_success": True}
    cube = np.array(target, dtype=np.float64)
    r = combined_reward(info, cube, target, shaping_weight=0.01)
    assert r == pytest.approx(1.0, abs=1e-9)
