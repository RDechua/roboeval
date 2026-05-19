"""Tests for the trained-residual Policy adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from roboeval.residual import ResidualCompositePolicy, ResidualCompositor


class _ConstantBasePolicy:
    """Minimal Policy stub returning a fixed action."""

    def __init__(self, value: float, action_dim: int = 14):
        self.policy_id = "mock_base"
        self.device = "cpu"
        self._value = value
        self._dim = action_dim
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1

    def select_action(
        self, observation: Mapping[str, object]
    ) -> npt.NDArray[np.float32]:
        del observation
        return np.full(self._dim, self._value, dtype=np.float32)


class _FakePPO:
    """Stub matching the SB3 ``predict`` signature for testing."""

    def __init__(self, residual_value: float, action_dim: int = 14):
        self._value = residual_value
        self._dim = action_dim
        self.predict_calls = 0
        self.last_deterministic: bool | None = None

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[npt.NDArray[Any], Any]:
        del observation, state, episode_start
        self.predict_calls += 1
        self.last_deterministic = deterministic
        return np.full(self._dim, self._value, dtype=np.float32), None


def _make_composite(
    *, base_value: float, residual_value: float, alpha_init: float = 0.1
) -> tuple[_ConstantBasePolicy, _FakePPO, ResidualCompositePolicy]:
    base = _ConstantBasePolicy(value=base_value)
    ppo = _FakePPO(residual_value=residual_value)
    compositor = ResidualCompositor(alpha_init=alpha_init)
    composite = ResidualCompositePolicy(
        base_policy=base, residual_model=ppo, compositor=compositor
    )
    return base, ppo, composite


def test_composite_policy_id_wraps_base_id():
    _b, _p, composite = _make_composite(base_value=0.0, residual_value=0.0)
    assert composite.policy_id == "residual(mock_base)"


def test_composite_device_inherits_from_base():
    _b, _p, composite = _make_composite(base_value=0.0, residual_value=0.0)
    assert composite.device == "cpu"


def test_composite_reset_delegates_to_base():
    base, _p, composite = _make_composite(base_value=0.0, residual_value=0.0)
    composite.reset()
    composite.reset()
    assert base.reset_calls == 2


def test_composite_action_equals_base_plus_alpha_residual():
    """alpha=0.5, base=0.3, residual=0.4 -> 0.3 + 0.5*0.4 = 0.5."""
    _b, _p, composite = _make_composite(
        base_value=0.3, residual_value=0.4, alpha_init=0.5
    )
    out = composite.select_action({"obs": "ignored"})
    assert out == pytest.approx(np.full(14, 0.5, dtype=np.float32), abs=1e-6)


def test_composite_action_clamps_to_action_space():
    """base=0.9 + 0.5*1.0 = 1.4 -> clamps to 1.0."""
    _b, _p, composite = _make_composite(
        base_value=0.9, residual_value=1.0, alpha_init=0.5
    )
    out = composite.select_action({"obs": "ignored"})
    assert out == pytest.approx(np.full(14, 1.0, dtype=np.float32), abs=1e-6)
    assert out.max() <= 1.0
    assert out.min() >= -1.0


def test_composite_requests_deterministic_action_from_residual_by_default():
    """Eval should use deterministic residual to avoid inflated TSR from luck."""
    _b, ppo, composite = _make_composite(base_value=0.0, residual_value=0.0)
    composite.select_action({"obs": "ignored"})
    assert ppo.last_deterministic is True


def test_composite_respects_non_deterministic_flag():
    base = _ConstantBasePolicy(value=0.0)
    ppo = _FakePPO(residual_value=0.0)
    compositor = ResidualCompositor(alpha_init=0.1)
    composite = ResidualCompositePolicy(
        base_policy=base,
        residual_model=ppo,
        compositor=compositor,
        deterministic=False,
    )
    composite.select_action({"obs": "ignored"})
    assert ppo.last_deterministic is False


def test_composite_select_action_dtype_is_float32():
    """RolloutResult and gym env both expect float32 actions."""
    _b, _p, composite = _make_composite(base_value=0.3, residual_value=0.4)
    out = composite.select_action({"obs": "ignored"})
    assert out.dtype == np.float32


def test_composite_calls_both_policies_per_step():
    base, ppo, composite = _make_composite(base_value=0.0, residual_value=0.0)
    for _ in range(5):
        composite.select_action({"obs": "ignored"})
    assert ppo.predict_calls == 5


def test_composite_does_not_call_residual_during_reset():
    """Reset should only affect the base policy, not the residual model."""
    _b, ppo, composite = _make_composite(base_value=0.0, residual_value=0.0)
    composite.reset()
    composite.reset()
    assert ppo.predict_calls == 0


def test_composite_passes_flat_obs_when_obs_builder_set():
    """Regression: PPO trained on flat Box obs; eval-time must rebuild it.

    The trained-residual SB3 model asserts ``obs_to_tensor`` sees the
    same shape it was trained on. Passing the raw gym-aloha Dict
    crashes with ``The observation provided is a dict but the obs
    space is Box(-inf, inf, (35,), float32)`` — the bug that surfaced
    on the first +5cm sparse residual eval.
    """

    class _RecordingPPO:
        def __init__(self) -> None:
            self.last_observation: Any = None

        def predict(
            self,
            observation: Any,
            state: Any = None,
            episode_start: Any = None,
            deterministic: bool = False,
        ) -> tuple[npt.NDArray[Any], Any]:
            del state, episode_start, deterministic
            self.last_observation = observation
            return np.zeros(14, dtype=np.float32), None

    base = _ConstantBasePolicy(value=0.2)
    ppo = _RecordingPPO()
    compositor = ResidualCompositor(alpha_init=0.05)

    captured_base_actions: list[Any] = []

    def _obs_builder(
        obs_dict: Mapping[str, object], base_action: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        del obs_dict
        captured_base_actions.append(base_action.copy())
        return np.full(35, 0.7, dtype=np.float32)

    composite = ResidualCompositePolicy(
        base_policy=base,
        residual_model=ppo,
        compositor=compositor,
        obs_builder=_obs_builder,
    )
    out = composite.select_action({"agent_pos": np.zeros(14)})

    # The SB3 mock saw the flat (35,) float32 obs from the builder,
    # not the raw Dict.
    assert isinstance(ppo.last_observation, np.ndarray)
    assert ppo.last_observation.shape == (35,)
    assert ppo.last_observation.dtype == np.float32
    assert float(ppo.last_observation[0]) == pytest.approx(0.7)

    # The builder received the base action (so PPO sees what the
    # wrapper showed it at training time).
    assert len(captured_base_actions) == 1
    assert captured_base_actions[0] == pytest.approx(np.full(14, 0.2, dtype=np.float32))

    # Composition still returns a float32 action.
    assert out.dtype == np.float32
    assert out.shape == (14,)


def test_composite_passes_observation_through_when_no_obs_builder():
    """Backward-compat: with obs_builder=None, predict sees the raw obs."""

    class _RecordingPPO:
        def __init__(self) -> None:
            self.last_observation: Any = None

        def predict(
            self,
            observation: Any,
            state: Any = None,
            episode_start: Any = None,
            deterministic: bool = False,
        ) -> tuple[npt.NDArray[Any], Any]:
            del state, episode_start, deterministic
            self.last_observation = observation
            return np.zeros(14, dtype=np.float32), None

    base = _ConstantBasePolicy(value=0.0)
    ppo = _RecordingPPO()
    compositor = ResidualCompositor(alpha_init=0.05)
    composite = ResidualCompositePolicy(
        base_policy=base, residual_model=ppo, compositor=compositor
    )
    obs = {"agent_pos": np.zeros(14), "sentinel": "passthrough"}
    composite.select_action(obs)
    assert ppo.last_observation is obs
