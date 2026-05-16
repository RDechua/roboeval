"""Test the W&B logger with mode='disabled' so no network/auth is touched."""

from __future__ import annotations

import pytest

from roboeval.evaluation.logger import wandb_run
from roboeval.evaluation.types import RolloutResult, aggregate


def _rollout(idx, success):
    return RolloutResult(
        seed_group=0,
        rollout_idx=idx,
        episode_seed=idx,
        success=success,
        success_custom=success,
        success_step=10 if success else None,
        n_steps=10,
        max_reward=4 if success else 0,
        terminated=success,
        truncated=not success,
        wall_time_s=0.1,
        final_cube_z=0.1,
        final_cube_x=0.0,
        final_cube_y=0.0,
        final_cube_xy_dist=0.0,
    )


def test_wandb_run_disabled_mode_does_not_raise(tmp_path, monkeypatch):
    pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))

    rollouts = [_rollout(i, success=(i < 3)) for i in range(5)]
    eval_result = aggregate(rollouts, policy_id="p", env_id="e")

    with wandb_run(
        project="roboeval-test",
        name="test_disabled",
        config={"policy_id": "p", "env_id": "e", "n_rollouts": 5},
        tags=["test"],
        mode="disabled",
    ) as handle:
        for r in rollouts:
            handle.log_rollout(r)
        handle.log_summary(eval_result)
        handle.log_distribution({"success": 3, "timeout": 2})
        # disabled-mode wandb sets url to None; nothing to assert beyond no-raise.
        assert handle.url is None or isinstance(handle.url, str)
        # run_id may be a string (synthetic id) or None depending on the
        # wandb version; the only contract is "doesn't raise".
        assert handle.run_id is None or isinstance(handle.run_id, str)
