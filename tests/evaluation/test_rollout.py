"""Unit tests for run_rollout with a mock policy and mock env.

The mock env never imports gym_aloha — we bypass the cube-state accessor
via the ``cube_state_fn`` parameter so the rollout loop never reaches
into ``env.unwrapped._env.physics``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from roboeval.envs.success import SuccessCriterion, TransferCubeSuccessDetector
from roboeval.evaluation.rollout import run_rollout


class MockEnv(gym.Env[Any, Any]):
    """Single-step-success env: declares is_success after `success_after_n` steps."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, action_dim=14, success_after_n=5, max_steps=20):
        agent_pos_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float64
        )
        self.observation_space = spaces.Dict({"agent_pos": agent_pos_space})
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self._success_after_n = success_after_n
        self._max_steps = max_steps
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        obs = {"agent_pos": np.zeros(14, dtype=np.float64)}
        return obs, {"is_success": False}

    def step(self, action):
        self._step += 1
        obs = {"agent_pos": np.zeros(14, dtype=np.float64)}
        reward = 4 if self._step >= self._success_after_n else 0
        terminated = self._step >= self._success_after_n
        truncated = self._step >= self._max_steps and not terminated
        info = {"is_success": terminated}
        return obs, reward, terminated, truncated, info


class MockPolicy:
    """No-op policy returning a fixed zero action."""

    policy_id = "mock"
    device = "cpu"

    def __init__(self, action_dim=14):
        self._action_dim = action_dim
        self.n_reset_calls = 0

    def reset(self):
        self.n_reset_calls += 1

    def select_action(
        self, observation: Mapping[str, object]
    ) -> npt.NDArray[np.float32]:
        del observation
        return np.zeros(self._action_dim, dtype=np.float32)


def _fake_cube_state(env):
    """Pretend the cube is on the table at the origin throughout."""
    del env
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def test_rollout_records_native_success_and_step():
    env = MockEnv(success_after_n=5, max_steps=20)
    policy = MockPolicy()
    detector = TransferCubeSuccessDetector(SuccessCriterion())
    result = run_rollout(
        env=env,
        policy=policy,
        success_detector=detector,
        seed_group=0,
        rollout_idx=3,
        episode_seed=123,
        max_steps=20,
        cube_state_fn=_fake_cube_state,
    )
    assert result.success is True
    assert result.success_custom is False  # cube at z=0, never hits z>0.05
    assert result.success_step == 5
    assert result.n_steps == 5
    assert result.max_reward == 4
    assert result.terminated is True
    assert result.truncated is False
    assert result.seed_group == 0
    assert result.rollout_idx == 3
    assert result.episode_seed == 123
    assert policy.n_reset_calls == 1


def test_rollout_truncation_records_truncated_true():
    env = MockEnv(success_after_n=100, max_steps=20)  # never succeeds
    policy = MockPolicy()
    detector = TransferCubeSuccessDetector(SuccessCriterion())
    result = run_rollout(
        env=env,
        policy=policy,
        success_detector=detector,
        seed_group=1,
        rollout_idx=0,
        episode_seed=0,
        max_steps=20,
        cube_state_fn=_fake_cube_state,
    )
    assert result.success is False
    assert result.truncated is True
    assert result.terminated is False
    assert result.success_step is None
    assert result.n_steps == 20


def test_rollout_custom_success_detected_via_high_cube():
    env = MockEnv(success_after_n=100, max_steps=20)

    def high_cube(env):
        return np.array([0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    detector = TransferCubeSuccessDetector(SuccessCriterion(dwell_steps=3))
    result = run_rollout(
        env=env,
        policy=MockPolicy(),
        success_detector=detector,
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=20,
        cube_state_fn=high_cube,
    )
    assert result.success_custom is True
    assert result.success_step == 3  # 1-based at first detection (dwell=3 → step 3)
