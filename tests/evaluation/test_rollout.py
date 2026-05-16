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
import pytest
from gymnasium import spaces

from roboeval.envs.success import SuccessCriterion, TransferCubeSuccessDetector
from roboeval.evaluation.rollout import run_rollout

# SuccessCriterion has no defaults; this helper rebuilds the old Week-2
# placeholder behaviour so the mock-env rollout tests stay readable.
_TEST_CRITERION_DEFAULTS = {
    "z_threshold_m": 0.05,
    "xy_tolerance_m": 0.05,
    "dwell_steps": 5,
    "target_xy": (0.0, 0.0),
}


def _crit(**overrides):
    return SuccessCriterion(**{**_TEST_CRITERION_DEFAULTS, **overrides})


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
    detector = TransferCubeSuccessDetector(_crit())
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
    detector = TransferCubeSuccessDetector(_crit())
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


class _BadPolicy:
    """Policy that returns a configurable bad action — used to trip safety asserts."""

    policy_id = "bad"
    device = "cpu"

    def __init__(self, action):
        self._action = action
        self.n_reset_calls = 0

    def reset(self):
        self.n_reset_calls += 1

    def select_action(self, observation):
        del observation
        return self._action


def test_rollout_raises_on_nan_action():
    env = MockEnv(success_after_n=100, max_steps=20)
    bad_action = np.full(14, np.nan, dtype=np.float32)
    with pytest.raises(RuntimeError, match="non-finite action"):
        run_rollout(
            env=env,
            policy=_BadPolicy(bad_action),
            success_detector=TransferCubeSuccessDetector(_crit()),
            seed_group=0,
            rollout_idx=0,
            episode_seed=0,
            max_steps=20,
            cube_state_fn=_fake_cube_state,
        )


def test_rollout_raises_on_inf_action():
    env = MockEnv(success_after_n=100, max_steps=20)
    bad_action = np.full(14, np.inf, dtype=np.float32)
    with pytest.raises(RuntimeError, match="non-finite action"):
        run_rollout(
            env=env,
            policy=_BadPolicy(bad_action),
            success_detector=TransferCubeSuccessDetector(_crit()),
            seed_group=0,
            rollout_idx=0,
            episode_seed=0,
            max_steps=20,
            cube_state_fn=_fake_cube_state,
        )


def test_rollout_raises_on_out_of_bound_action():
    env = MockEnv(success_after_n=100, max_steps=20)
    bad_action = np.full(14, 1e6, dtype=np.float32)
    with pytest.raises(RuntimeError, match="out-of-bound action"):
        run_rollout(
            env=env,
            policy=_BadPolicy(bad_action),
            success_detector=TransferCubeSuccessDetector(_crit()),
            seed_group=0,
            rollout_idx=0,
            episode_seed=0,
            max_steps=20,
            cube_state_fn=_fake_cube_state,
        )


def test_rollout_accepts_normal_actions_within_bounds():
    """Slightly-out-of-box but within sanity ceiling should still be accepted."""
    env = MockEnv(success_after_n=5, max_steps=20)
    ok_action = np.full(14, 5.0, dtype=np.float32)  # > 1.0 box, << 100.0 ceiling
    result = run_rollout(
        env=env,
        policy=_BadPolicy(ok_action),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=20,
        cube_state_fn=_fake_cube_state,
    )
    assert result.n_steps == 5


def test_rollout_custom_success_detected_via_high_cube():
    env = MockEnv(success_after_n=100, max_steps=20)

    def high_cube(env):
        return np.array([0.0, 0.0, 0.10, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    detector = TransferCubeSuccessDetector(_crit(dwell_steps=3))
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


# --- Trajectory-aggregate tests (Week 5) -----------------------------------


class _AlternatingSignPolicy:
    """Returns +1, -1, +1, -1, ... — every step is a sign flip across all dims."""

    policy_id = "alternating"
    device = "cpu"

    def __init__(self, action_dim: int = 14):
        self._action_dim = action_dim
        self._t = 0

    def reset(self) -> None:
        self._t = 0

    def select_action(self, observation):
        del observation
        sign = 1.0 if self._t % 2 == 0 else -1.0
        self._t += 1
        return np.full(self._action_dim, sign, dtype=np.float32)


def test_rollout_action_sign_flip_rate_zero_for_constant_action():
    env = MockEnv(success_after_n=100, max_steps=10)
    result = run_rollout(
        env=env,
        policy=MockPolicy(),  # constant zero action
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=10,
        cube_state_fn=_fake_cube_state,
    )
    assert result.action_sign_flip_rate == 0.0


def test_rollout_action_sign_flip_rate_one_for_alternating_action():
    env = MockEnv(success_after_n=100, max_steps=10)
    result = run_rollout(
        env=env,
        policy=_AlternatingSignPolicy(),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=10,
        cube_state_fn=_fake_cube_state,
    )
    # Every step except the first is a sign flip across every dim.
    assert result.action_sign_flip_rate == pytest.approx(1.0)


def test_rollout_records_contact_made_when_contact_fn_fires():
    env = MockEnv(success_after_n=100, max_steps=10)
    calls = {"n": 0}

    def fake_contact(env):
        del env
        calls["n"] += 1
        return calls["n"] == 5  # fire once mid-episode

    result = run_rollout(
        env=env,
        policy=MockPolicy(),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=10,
        cube_state_fn=_fake_cube_state,
        contact_fn=fake_contact,
    )
    assert result.contact_made is True


def test_rollout_contact_made_false_when_contact_fn_never_fires():
    env = MockEnv(success_after_n=100, max_steps=10)
    result = run_rollout(
        env=env,
        policy=MockPolicy(),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=10,
        cube_state_fn=_fake_cube_state,
        contact_fn=lambda env: False,
    )
    assert result.contact_made is False


def test_rollout_terminal_eef_xy_distance_uses_closer_gripper():
    env = MockEnv(success_after_n=5, max_steps=10)

    def cube_at_origin(env):
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def grippers_left_far_right_near(env):
        del env
        left = np.array([1.0, 0.0], dtype=np.float64)
        right = np.array([0.03, 0.04], dtype=np.float64)  # distance 0.05
        return left, right

    result = run_rollout(
        env=env,
        policy=MockPolicy(),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=10,
        cube_state_fn=cube_at_origin,
        gripper_xy_fn=grippers_left_far_right_near,
    )
    assert result.terminal_eef_xy_distance_m == pytest.approx(0.05, abs=1e-6)


def test_rollout_terminal_eef_xy_distance_is_none_when_accessor_returns_none():
    env = MockEnv(success_after_n=5, max_steps=10)
    result = run_rollout(
        env=env,
        policy=MockPolicy(),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=10,
        cube_state_fn=_fake_cube_state,
        gripper_xy_fn=lambda env: None,
    )
    assert result.terminal_eef_xy_distance_m is None


def test_rollout_last_50_step_cube_displacement_short_window():
    # Run for 6 steps with cube moving 0.01 m in +y per step; window collapses
    # to the full episode (6 steps < 50), so the displacement = 5 * 0.01 = 0.05.
    env = MockEnv(success_after_n=100, max_steps=6)
    step_counter = {"n": 0}

    def drifting_cube(env):
        del env
        y = 0.01 * step_counter["n"]
        step_counter["n"] += 1
        return np.array([0.0, y, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    result = run_rollout(
        env=env,
        policy=MockPolicy(),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=6,
        cube_state_fn=drifting_cube,
    )
    # 7 cube samples recorded (1 pre-step + 6 post-step), so displacement
    # spans steps 0..6, i.e. 6 * 0.01 = 0.06 m.
    assert result.last_50_step_cube_displacement_m == pytest.approx(0.06, abs=1e-6)


def test_rollout_last_50_step_cube_displacement_zero_for_static_cube():
    env = MockEnv(success_after_n=100, max_steps=10)
    result = run_rollout(
        env=env,
        policy=MockPolicy(),
        success_detector=TransferCubeSuccessDetector(_crit()),
        seed_group=0,
        rollout_idx=0,
        episode_seed=0,
        max_steps=10,
        cube_state_fn=_fake_cube_state,
    )
    assert result.last_50_step_cube_displacement_m == 0.0
