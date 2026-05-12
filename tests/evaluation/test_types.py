"""Unit tests for the result schemas and aggregate()."""

from __future__ import annotations

import pytest

from roboeval.evaluation.types import RolloutResult, aggregate


def _rollout(seed_group, idx, success, success_custom=False, step=None):
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
        final_cube_y=0.0,
        final_cube_xy_dist=0.01,
    )


def test_aggregate_empty_raises():
    with pytest.raises(ValueError, match=r"at least one rollout"):
        aggregate([], policy_id="p", env_id="e")


def test_aggregate_single_seed_zero_std():
    rollouts = [_rollout(0, i, success=(i < 4), step=10) for i in range(10)]
    result = aggregate(rollouts, policy_id="p", env_id="e")
    assert result.n_rollouts == 10
    assert result.n_seed_groups == 1
    assert result.mean_tsr == pytest.approx(0.4)
    assert result.std_tsr == 0.0  # single group → pstdev = 0
    assert result.median_tts == 10.0


def test_aggregate_three_seeds_std_across_groups():
    # group 0: 4/10 success, group 1: 6/10, group 2: 8/10 → mean 0.6
    rollouts = []
    for seed_group, n_success in [(0, 4), (1, 6), (2, 8)]:
        for i in range(10):
            rollouts.append(_rollout(seed_group, i, success=(i < n_success)))
    result = aggregate(rollouts, policy_id="p", env_id="e")
    assert result.n_seed_groups == 3
    assert result.mean_tsr == pytest.approx(0.6)
    # per-group TSRs are 0.4, 0.6, 0.8 → pstdev ≈ 0.1633
    assert result.std_tsr == pytest.approx(0.1633, abs=1e-3)
    assert result.per_seed_tsr == pytest.approx((0.4, 0.6, 0.8))


def test_aggregate_median_tts_only_over_successes():
    rollouts = [
        _rollout(0, 0, success=True, step=100),
        _rollout(0, 1, success=True, step=200),
        _rollout(0, 2, success=False),  # excluded from median TTS
        _rollout(0, 3, success=True, step=150),
    ]
    result = aggregate(rollouts, policy_id="p", env_id="e")
    assert result.median_tts == 150.0


def test_aggregate_no_successes_median_tts_is_none():
    rollouts = [_rollout(0, i, success=False) for i in range(5)]
    result = aggregate(rollouts, policy_id="p", env_id="e")
    assert result.median_tts is None
    assert result.mean_tsr == 0.0
