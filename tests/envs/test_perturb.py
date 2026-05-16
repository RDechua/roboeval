"""Unit tests for SpatialShiftWrapper and the perturbation factory.

The tests use a tiny mock env that mirrors the gym_aloha attribute path
``env.unwrapped._env.physics.data.qpos`` without requiring real
mujoco/dm_control. The wrapper writes to ``qpos[16:18]`` per the
PRD-defined cube qpos slice.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from roboeval.envs.perturb import (
    SpatialShiftWrapper,
    _cube_xy_indices,
    make_perturbed_env,
)


class _MockAlohaLikeEnv(gym.Env[Any, Any]):
    """Mirrors the env.unwrapped._env.physics.data.qpos attribute path."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, initial_cube_xy: tuple[float, float] = (0.10, 0.20)) -> None:
        self.observation_space = spaces.Dict(
            {"agent_pos": spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float64)}
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        self._initial_cube_xy = initial_cube_xy
        self.forward_call_count = 0

        # Build the dm_control-shaped attribute tree the wrapper walks.
        qpos = np.zeros(23, dtype=np.float64)
        physics = SimpleNamespace(
            data=SimpleNamespace(qpos=qpos),
            forward=self._record_forward,
        )
        self._env_ = SimpleNamespace(physics=physics)

    @property
    def _env(self) -> SimpleNamespace:
        return self._env_

    def _record_forward(self) -> None:
        self.forward_call_count += 1

    def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
        super().reset(seed=seed)
        # Match what gym_aloha does: write the initial cube pose to qpos.
        ix, iy = _cube_xy_indices()
        self._env_.physics.data.qpos[ix] = self._initial_cube_xy[0]
        self._env_.physics.data.qpos[iy] = self._initial_cube_xy[1]
        return {"agent_pos": np.zeros(14, dtype=np.float64)}, {"is_success": False}

    def step(self, action):  # type: ignore[no-untyped-def]
        del action
        return (
            {"agent_pos": np.zeros(14, dtype=np.float64)},
            0,
            False,
            False,
            {"is_success": False},
        )


def test_spatial_shift_writes_to_correct_qpos_indices():
    base = _MockAlohaLikeEnv(initial_cube_xy=(0.10, 0.20))
    wrapped = SpatialShiftWrapper(base, dx_m=0.05, dy_m=-0.02)
    wrapped.reset(seed=0)
    ix, iy = _cube_xy_indices()
    # Base reset wrote (0.10, 0.20); wrapper added (0.05, -0.02).
    assert base._env.physics.data.qpos[ix] == pytest.approx(0.15)
    assert base._env.physics.data.qpos[iy] == pytest.approx(0.18)


def test_spatial_shift_calls_forward_to_refresh_kinematics():
    base = _MockAlohaLikeEnv()
    wrapped = SpatialShiftWrapper(base, dx_m=0.0, dy_m=0.03)
    wrapped.reset(seed=0)
    assert base.forward_call_count == 1, "physics.forward must run after qpos edit"


def test_spatial_shift_is_deterministic_across_resets():
    base = _MockAlohaLikeEnv(initial_cube_xy=(0.0, 0.5))
    wrapped = SpatialShiftWrapper(base, dx_m=0.0, dy_m=0.05)
    ix, iy = _cube_xy_indices()
    for _ in range(3):
        wrapped.reset(seed=42)
        assert base._env.physics.data.qpos[ix] == pytest.approx(0.0)
        assert base._env.physics.data.qpos[iy] == pytest.approx(0.55)


def test_spatial_shift_zero_delta_is_a_noop():
    base = _MockAlohaLikeEnv(initial_cube_xy=(0.10, 0.20))
    wrapped = SpatialShiftWrapper(base, dx_m=0.0, dy_m=0.0)
    wrapped.reset(seed=0)
    ix, iy = _cube_xy_indices()
    assert base._env.physics.data.qpos[ix] == pytest.approx(0.10)
    assert base._env.physics.data.qpos[iy] == pytest.approx(0.20)


def test_spatial_shift_exposes_deltas_for_logging():
    wrapped = SpatialShiftWrapper(_MockAlohaLikeEnv(), dx_m=0.01, dy_m=0.03)
    assert wrapped.dx_m == 0.01
    assert wrapped.dy_m == 0.03


def test_make_perturbed_env_dispatches_to_spatial():
    base = _MockAlohaLikeEnv(initial_cube_xy=(0.0, 0.5))
    out = make_perturbed_env(base, kind="spatial", dx_m=0.0, dy_m=0.02)
    assert isinstance(out, SpatialShiftWrapper)
    assert out.dy_m == pytest.approx(0.02)


def test_make_perturbed_env_unknown_kind_raises():
    base = _MockAlohaLikeEnv()
    with pytest.raises(ValueError, match=r"unknown perturbation kind 'jumble'"):
        make_perturbed_env(base, kind="jumble")


def test_make_perturbed_env_unknown_kind_lists_supported():
    base = _MockAlohaLikeEnv()
    with pytest.raises(ValueError, match=r"supported: \[.*spatial.*\]"):
        make_perturbed_env(base, kind="xyz")


@pytest.mark.parametrize("kind", ["visual", "dynamic", "temporal"])
def test_make_perturbed_env_reserved_kinds_raise_not_implemented(kind: str):
    base = _MockAlohaLikeEnv()
    with pytest.raises(NotImplementedError, match=r"Week 4"):
        make_perturbed_env(base, kind=kind)


def test_make_perturbed_env_spatial_defaults_to_zero_shift():
    """If a config omits dx_m/dy_m, the wrapper applies a zero shift (no error)."""
    base = _MockAlohaLikeEnv(initial_cube_xy=(0.10, 0.20))
    out = make_perturbed_env(base, kind="spatial")  # no dx/dy params
    assert isinstance(out, SpatialShiftWrapper)
    out.reset(seed=0)
    ix, iy = _cube_xy_indices()
    assert base._env.physics.data.qpos[ix] == pytest.approx(0.10)
    assert base._env.physics.data.qpos[iy] == pytest.approx(0.20)
