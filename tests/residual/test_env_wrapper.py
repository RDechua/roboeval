"""Unit tests for ResidualEnvWrapper.

The wrapper exposes a flat Box observation to PPO (agent_pos +
cube_state + base_action + optional feature_extractor output) while
internally orchestrating the base-policy + residual composition.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium import spaces

from roboeval.residual import (
    ResidualCompositor,
    ResidualEnvWrapper,
    zero_feature_extractor,
)


class _MockEnv(gym.Env[Any, Any]):
    """Records every action the env actually executes."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, action_dim: int = 14, success_after: int = 100) -> None:
        agent_pos = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float64)
        self.observation_space = spaces.Dict({"agent_pos": agent_pos})
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.received: list[np.ndarray] = []
        self._step = 0
        self._success_after = success_after

    def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
        super().reset(seed=seed)
        self._step = 0
        self.received.clear()
        return {"agent_pos": np.zeros(14, dtype=np.float64)}, {"is_success": False}

    def step(self, action):  # type: ignore[no-untyped-def]
        self.received.append(np.asarray(action, dtype=np.float32).copy())
        self._step += 1
        success = self._step >= self._success_after
        return (
            {"agent_pos": np.zeros(14, dtype=np.float64)},
            0.0,
            success,
            False,
            {"is_success": success},
        )


class _ConstantBasePolicy:
    policy_id = "mock_base"
    device = "cpu"

    def __init__(self, action_dim: int = 14, value: float = 0.3) -> None:
        self._dim = action_dim
        self._value = value
        self.reset_calls = 0
        self.select_action_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(
        self, observation: Mapping[str, object]
    ) -> npt.NDArray[np.float32]:
        del observation
        self.select_action_calls += 1
        return np.full(self._dim, self._value, dtype=np.float32)


def _flat_cube_xy(_env: gym.Env[Any, Any]) -> npt.NDArray[np.float64]:
    return np.array([0.1, 0.2, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def _make_wrapper(
    *,
    alpha_init: float = 0.1,
    base_value: float = 0.3,
) -> tuple[_MockEnv, _ConstantBasePolicy, ResidualEnvWrapper]:
    env = _MockEnv()
    base = _ConstantBasePolicy(value=base_value)
    compositor = ResidualCompositor(alpha_init=alpha_init)
    wrapper = ResidualEnvWrapper(
        env=env,
        base_policy=base,
        compositor=compositor,
        reward_fn=lambda info, cube_xy: 1.0 if info.get("is_success") else 0.0,
        feature_extractor=zero_feature_extractor,
        cube_state_fn=_flat_cube_xy,
    )
    return env, base, wrapper


def test_wrapper_observation_space_is_flat_box():
    """SB3's DummyVecEnv rejects nested Dicts; flat Box is the v1 contract."""
    _env, _base, wrapper = _make_wrapper()
    assert isinstance(wrapper.observation_space, spaces.Box)
    # agent_pos(14) + cube_state(7) + base_action(14) + zero features(0) = 35.
    assert wrapper.observation_space.shape == (35,)
    assert wrapper.observation_space.dtype == np.float32


def test_wrapper_reset_resets_base_policy():
    _env, base, wrapper = _make_wrapper()
    base.reset_calls = 0  # _infer_feature_dim ran a reset already
    wrapper.reset(seed=0)
    assert base.reset_calls == 1


def test_wrapper_reset_returns_flat_obs_with_correct_components():
    _env, _base, wrapper = _make_wrapper(base_value=0.3)
    flat_obs, _info = wrapper.reset(seed=0)
    # Layout: agent_pos(14)=0 ++ cube(7)=[.1,.2,.05,1,0,0,0] ++ base(14)=.3 ++ feat(0)
    assert flat_obs.shape == (35,)
    assert flat_obs.dtype == np.float32
    assert flat_obs[:14] == pytest.approx(np.zeros(14, dtype=np.float32))
    assert flat_obs[14:21] == pytest.approx(
        np.array([0.1, 0.2, 0.05, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )
    assert flat_obs[21:35] == pytest.approx(np.full(14, 0.3, dtype=np.float32))


def test_wrapper_step_composes_base_plus_alpha_residual():
    """clamp(0.3 + 0.5 * 0.4, -1, 1) = 0.5 → env should receive 0.5."""
    env, _base, wrapper = _make_wrapper(alpha_init=0.5, base_value=0.3)
    wrapper.reset(seed=0)
    residual = np.full(14, 0.4, dtype=np.float32)
    wrapper.step(residual)
    received = env.received[0]
    assert received == pytest.approx(np.full(14, 0.5, dtype=np.float32), abs=1e-6)


def test_wrapper_clamps_composed_action_to_action_space():
    env, _base, wrapper = _make_wrapper(alpha_init=0.5, base_value=0.9)
    wrapper.reset(seed=0)
    residual = np.full(14, 1.0, dtype=np.float32)
    wrapper.step(residual)
    assert env.received[0] == pytest.approx(np.full(14, 1.0, dtype=np.float32))


def test_wrapper_replaces_reward_with_custom_reward_fn():
    env = _MockEnv(success_after=1)
    base = _ConstantBasePolicy()
    compositor = ResidualCompositor(alpha_init=0.1)
    wrapper = ResidualEnvWrapper(
        env=env,
        base_policy=base,
        compositor=compositor,
        reward_fn=lambda info, cube_xy: 7.0,
        cube_state_fn=_flat_cube_xy,
    )
    wrapper.reset(seed=0)
    _obs, reward, _term, _trunc, _info = wrapper.step(np.zeros(14, dtype=np.float32))
    assert reward == 7.0


def test_wrapper_passes_info_through_unchanged():
    env = _MockEnv(success_after=1)
    base = _ConstantBasePolicy()
    compositor = ResidualCompositor(alpha_init=0.1)
    wrapper = ResidualEnvWrapper(
        env=env,
        base_policy=base,
        compositor=compositor,
        reward_fn=lambda info, cube_xy: 0.0,
        cube_state_fn=_flat_cube_xy,
    )
    wrapper.reset(seed=0)
    _obs, _r, term, _trunc, info = wrapper.step(np.zeros(14, dtype=np.float32))
    assert info["is_success"] is True
    assert term is True


def test_wrapper_step_before_reset_raises():
    _env, _base, wrapper = _make_wrapper()
    # Construction probed reset via _infer_feature_dim; explicitly clear the cache
    # to simulate "user never called reset on the wrapper".
    wrapper._cached_base_action = None
    with pytest.raises(RuntimeError, match="reset"):
        wrapper.step(np.zeros(14, dtype=np.float32))


def test_wrapper_calls_base_select_action_once_per_env_step():
    """ACT-style chunked policies must not be called twice per step (chunk-pointer)."""
    _env, base, wrapper = _make_wrapper()
    base.select_action_calls = 0
    wrapper.reset(seed=0)
    # reset() makes one call to seed the cache.
    calls_after_reset = base.select_action_calls
    assert calls_after_reset == 1
    wrapper.step(np.zeros(14, dtype=np.float32))
    wrapper.step(np.zeros(14, dtype=np.float32))
    wrapper.step(np.zeros(14, dtype=np.float32))
    # 3 step() calls add exactly 3 select_action calls (one per next-obs cache).
    assert base.select_action_calls == calls_after_reset + 3


def test_zero_feature_extractor_returns_empty_array():
    out = zero_feature_extractor({"anything": "ignored"})
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_wrapper_feature_extractor_extends_flat_obs():
    """A non-empty feature_extractor must concatenate onto the tail of flat_obs."""
    env = _MockEnv()
    base = _ConstantBasePolicy(value=0.2)
    compositor = ResidualCompositor(alpha_init=0.1)
    wrapper = ResidualEnvWrapper(
        env=env,
        base_policy=base,
        compositor=compositor,
        reward_fn=lambda info, cube_xy: 0.0,
        feature_extractor=lambda _o: np.array([1.5, -2.5], dtype=np.float32),
        cube_state_fn=_flat_cube_xy,
    )
    # 14 + 7 + 14 + 2 = 37
    assert wrapper.observation_space.shape == (37,)
    flat_obs, _info = wrapper.reset(seed=0)
    assert flat_obs.shape == (37,)
    assert flat_obs[-2:] == pytest.approx(np.array([1.5, -2.5], dtype=np.float32))
