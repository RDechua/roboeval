"""Gym wrapper that turns an ALOHA env into a residual-action env (PRD §8.2).

PPO sees the wrapper's action space as the **residual** action space
(same shape as the underlying env's action space). On every ``step``,
the wrapper:

1. Queries the frozen base policy for the per-step base action.
2. Combines base + residual via :class:`ResidualCompositor`.
3. Forwards the composed action to the underlying env.
4. Replaces the env's reward with the custom reward function (sparse
   success or shaped distance-to-goal, per PRD §8.3 Conditions B / C).

The wrapper is the *only* component aware of the base policy. The
residual MLP, the compositor, and the SB3 training loop are all
base-policy-agnostic. This keeps the policy choice (ACT in v1.0,
diffusion in v1.1) swappable from a single constructor argument.

Feature extraction
------------------
PRD §8.2 specifies the residual MLP input as
``(obs_features, base_action)``. The "obs_features" are the ACT
encoder's intermediate representation; extracting them requires hooking
into lerobot's ACT internals which is intentionally out of scope for
this v1 scaffold. The wrapper accepts a ``feature_extractor`` callable
so the caller chooses: a no-op (zero-width features, suitable for unit
tests and a simpler MLP), or a real ACT-encoder hook (Week 7).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch

from roboeval.envs.aloha import get_cube_state
from roboeval.policies.base import Policy
from roboeval.residual.policy import ResidualCompositor

CubeStateFn = Callable[[gym.Env[Any, Any]], npt.NDArray[np.float64]]
"""Accessor for the cube's 7-element qpos slice."""

FeatureExtractor = Callable[[Mapping[str, Any]], npt.NDArray[np.float32]]
"""Map an observation dict to a flat feature vector for the residual MLP."""

RewardFn = Callable[
    [Mapping[str, Any], npt.NDArray[np.float64]],
    float,
]
"""``reward_fn(info, cube_xy) -> reward``; constructed by the train loop."""


def zero_feature_extractor(_obs: Mapping[str, Any]) -> npt.NDArray[np.float32]:
    """No-op feature extractor: returns an empty (0-dim) feature vector.

    Useful for unit tests and for a simplified residual that conditions
    only on the base action. Pair with ``ResidualMLP(obs_feature_dim=0)``.
    """
    return np.zeros(0, dtype=np.float32)


class ResidualEnvWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Wrap an ALOHA env so PPO sees residual actions on top of a frozen base.

    The wrapper does **not** subclass ``Policy``; it is a gym wrapper.
    The residual policy itself is what PPO trains — SB3 sees the
    wrapped env as the training environment.

    Lifecycle per episode:

    * ``reset()``  — resets the underlying env AND calls
      ``base_policy.reset()`` so the base's internal chunk buffer
      restarts from step 0.
    * ``step(residual_action)`` — pulls a fresh base action from the
      base policy, composes via the compositor, forwards to the env,
      then replaces the env's reward via ``reward_fn``.

    Observations
    ------------
    The wrapper passes the env's native observation dict through
    unchanged. SB3's policy receives this dict; consumers needing a
    flat observation (e.g. SB3 ``MlpPolicy``) should compose this
    wrapper with a flattening wrapper.
    """

    def __init__(
        self,
        env: gym.Env[Any, Any],
        base_policy: Policy,
        compositor: ResidualCompositor,
        reward_fn: RewardFn,
        feature_extractor: FeatureExtractor = zero_feature_extractor,
        cube_state_fn: CubeStateFn = get_cube_state,
    ) -> None:
        """Wrap ``env`` with a base policy + compositor + reward shaper.

        Args:
            env: Underlying ALOHA env (typically already perturbed via
                :func:`roboeval.envs.perturb.make_perturbed_env`).
            base_policy: Frozen base policy (typically loaded via
                :func:`roboeval.policies.factory.load_policy`).
            compositor: :class:`ResidualCompositor` instance. Its alpha
                parameter is part of the wrapper's persistent state.
            reward_fn: ``reward_fn(info, cube_xy) -> reward`` callable.
                The train loop constructs this from
                :mod:`roboeval.residual.reward` and the eval config's
                target_xy + shaping_weight.
            feature_extractor: Maps the per-step obs to a flat feature
                vector for the residual MLP. Defaults to the zero-width
                extractor.
            cube_state_fn: Accessor for the cube's qpos slice.
        """
        super().__init__(env)
        self._base_policy = base_policy
        self._compositor = compositor
        self._reward_fn = reward_fn
        self._feature_extractor = feature_extractor
        self._cube_state_fn = cube_state_fn
        self._last_obs: Mapping[str, Any] | None = None

    @property
    def compositor(self) -> ResidualCompositor:
        """Expose the compositor so callers (eg checkpointing) can read alpha."""
        return self._compositor

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset env + base policy."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._base_policy.reset()
        self._last_obs = obs
        return obs, info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Forward (base + alpha * residual) to the env, swap reward in.

        Args:
            action: Residual action sampled by PPO. Shape matches the
                env's action space.

        Returns:
            ``(obs, reward, terminated, truncated, info)``. The reward
            is the custom :func:`combined_reward` output, NOT the env's
            sparse reward (which gym-aloha emits via ``info`` anyway).
        """
        if self._last_obs is None:
            raise RuntimeError(
                "ResidualEnvWrapper.step called before reset(); call reset first."
            )
        base_action_np = self._base_policy.select_action(self._last_obs)
        with torch.no_grad():
            base_t = torch.from_numpy(np.asarray(base_action_np, dtype=np.float32))
            residual_t = torch.from_numpy(np.asarray(action, dtype=np.float32))
            composed_t = self._compositor(base_t, residual_t)
        composed_np = composed_t.cpu().numpy().astype(np.float32)

        obs, _native_reward, terminated, truncated, info = self.env.step(composed_np)
        cube_xy = self._cube_state_fn(self.env)[:2]
        reward = float(self._reward_fn(info, cube_xy))
        self._last_obs = obs
        return obs, reward, bool(terminated), bool(truncated), info

    def extract_obs_features(self, obs: Mapping[str, Any]) -> npt.NDArray[np.float32]:
        """Public wrapper around the feature_extractor for symmetry with train code."""
        return self._feature_extractor(obs)


__all__ = [
    "FeatureExtractor",
    "ResidualEnvWrapper",
    "RewardFn",
    "zero_feature_extractor",
]
