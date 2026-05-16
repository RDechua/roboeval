"""Unit test for evaluate_policy aggregation across seed groups."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from roboeval.envs.success import SuccessCriterion, TransferCubeSuccessDetector
from roboeval.evaluation.loop import evaluate_policy


def _crit():
    # Reproduces the Week-2 placeholder defaults so this loop test reads
    # the same as before SuccessCriterion lost its dataclass defaults.
    return SuccessCriterion(
        z_threshold_m=0.05,
        xy_tolerance_m=0.05,
        dwell_steps=5,
        target_xy=(0.0, 0.0),
    )


class _DeterministicEnv(gym.Env[Any, Any]):
    """Env whose success rate depends on the seed via a hash."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self):
        self.observation_space = spaces.Dict(
            {"agent_pos": spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float64)}
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        self._step = 0
        self._will_succeed = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        # Deterministic: seeds 0..49 succeed; 100_003..100_052 fail; etc.
        self._will_succeed = (seed or 0) < 50
        return {"agent_pos": np.zeros(14, dtype=np.float64)}, {"is_success": False}

    def step(self, action):
        self._step += 1
        terminated = self._step >= 3 and self._will_succeed
        return (
            {"agent_pos": np.zeros(14, dtype=np.float64)},
            4 if terminated else 0,
            terminated,
            False,
            {"is_success": terminated},
        )


class _NoopPolicy:
    policy_id = "noop"
    device = "cpu"

    def reset(self):
        pass

    def select_action(
        self, observation: Mapping[str, object]
    ) -> npt.NDArray[np.float32]:
        del observation
        return np.zeros(14, dtype=np.float32)


def _fake_cube_state(env):
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def test_evaluate_policy_three_seed_groups():
    captured = []
    result = evaluate_policy(
        env_factory=_DeterministicEnv,
        policy=_NoopPolicy(),
        detector_factory=lambda: TransferCubeSuccessDetector(_crit()),
        seeds=[0, 1, 2],  # only seed_group=0 yields seeds < 50 → all succeed
        n_rollouts_per_seed=5,
        max_steps=10,
        policy_id="noop",
        env_id="mock",
        on_rollout=captured.append,
        cube_state_fn=_fake_cube_state,
    )
    assert len(captured) == 15
    assert result.n_seed_groups == 3
    assert result.n_rollouts == 15
    # Group 0: all 5 succeed; groups 1, 2: all fail.
    assert result.per_seed_tsr == (1.0, 0.0, 0.0)
    assert result.mean_tsr == 1.0 / 3
