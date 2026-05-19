"""Regression: train_residual plumbs log_std_init through to SB3 PPO policy_kwargs.

Doesn't actually run PPO (slow, requires gym-aloha). Instead patches
``stable_baselines3.PPO`` with a recording double, calls train_residual,
and asserts the constructor saw the expected policy_kwargs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
from gymnasium import spaces

pytest.importorskip("stable_baselines3")

from roboeval.residual import ResidualCompositor, train_residual  # noqa: E402
from roboeval.residual.env_wrapper import zero_feature_extractor  # noqa: E402


class _MockEnv(gym.Env[Any, Any]):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self) -> None:
        agent_pos = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float64)
        self.observation_space = spaces.Dict({"agent_pos": agent_pos})
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):  # type: ignore[no-untyped-def]
        super().reset(seed=seed)
        return {"agent_pos": np.zeros(14, dtype=np.float64)}, {"is_success": False}

    def step(self, action):  # type: ignore[no-untyped-def]
        del action
        return (
            {"agent_pos": np.zeros(14, dtype=np.float64)},
            0.0,
            False,
            False,
            {"is_success": False},
        )


class _ConstantBasePolicy:
    policy_id = "mock_base"
    device = "cpu"

    def reset(self) -> None: ...

    def select_action(
        self, observation: Mapping[str, object]
    ) -> npt.NDArray[np.float32]:
        del observation
        return np.zeros(14, dtype=np.float32)


def _run_train_with_mocked_ppo(
    *, log_std_init: float | None = None, tmp_path
) -> dict[str, Any]:
    base = _ConstantBasePolicy()
    compositor = ResidualCompositor(alpha_init=0.05)
    fake_model = MagicMock()
    with patch("stable_baselines3.PPO", return_value=fake_model) as ppo_ctor:
        kwargs: dict[str, Any] = {
            "base_env_factory": _MockEnv,
            "base_policy": base,
            "compositor": compositor,
            "reward_fn": lambda info, cube: 0.0,
            "output_dir": tmp_path,
            "total_timesteps": 1,
            "n_steps": 1,
            "batch_size": 1,
            "feature_extractor": zero_feature_extractor,
        }
        if log_std_init is not None:
            kwargs["log_std_init"] = log_std_init
        train_residual(**kwargs)
    ppo_ctor.assert_called_once()
    return dict(ppo_ctor.call_args.kwargs)


def test_train_residual_passes_log_std_init_to_ppo(tmp_path):
    call_kwargs = _run_train_with_mocked_ppo(log_std_init=-2.0, tmp_path=tmp_path)
    policy_kwargs = call_kwargs["policy_kwargs"]
    assert policy_kwargs["log_std_init"] == pytest.approx(-2.0)
    assert policy_kwargs["net_arch"] == [256, 256]


def test_train_residual_default_log_std_init_is_zero(tmp_path):
    """Default matches SB3's default behaviour (std=1.0)."""
    call_kwargs = _run_train_with_mocked_ppo(log_std_init=None, tmp_path=tmp_path)
    assert call_kwargs["policy_kwargs"]["log_std_init"] == 0.0
