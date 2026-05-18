"""Gym wrapper that turns an ALOHA env into a residual-action env (PRD §8.2).

PPO sees the wrapper's observation space as a **flat Box** containing
privileged sim state (joint qpos, cube qpos, last base action) so
SB3's ``MlpPolicy`` can consume it directly. The action space is the
**residual action space** (same shape and bounds as the underlying
env's). On every ``step``, the wrapper:

1. Composes the residual with the cached base action via
   :class:`ResidualCompositor`.
2. Forwards the composed action to the underlying env.
3. Queries the frozen base policy ONCE for the next-step base action
   (matters for chunked-action policies like ACT — calling
   ``select_action`` twice per env step would advance ACT's chunk
   pointer twice).
4. Builds the next flat observation and replaces the env's reward
   with the configured reward function.

The flat-obs design is a deliberate v1 deviation from the PRD §8.2
input spec ("obs_features from ACT encoder + base action"). The
deviation lets PPO actually train without a custom SB3 policy: SB3
won't accept gym-aloha's nested Dict observation. The Week-7 ACT-
encoder hook will be plugged in via the ``feature_extractor`` callable
and concatenated onto the flat obs.

Observation layout
------------------
``flat_obs = [agent_pos (14), cube_state (7), base_action (14),
              feature_extractor(obs) (variable)]``

For the default :func:`zero_feature_extractor` this is 35-dim. For an
ACT-encoder hook returning a 512-dim feature vector the flat obs grows
to 547-dim — backward-compatible because feature_extractor's output
is concatenated at the tail.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import torch
from gymnasium import spaces

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

_CUBE_STATE_DIM: int = 7
"""Length of the cube qpos slice (xyz + quat)."""


def zero_feature_extractor(_obs: Mapping[str, Any]) -> npt.NDArray[np.float32]:
    """No-op feature extractor: returns an empty (0-dim) feature vector.

    Default for v1: PPO conditions on the privileged sim state already
    in the flat obs (agent_pos + cube_state + base_action). The
    feature_extractor slot is reserved for the Week-7 ACT-encoder hook
    that returns real perceptual features.
    """
    return np.zeros(0, dtype=np.float32)


class ResidualEnvWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Wrap an ALOHA env so PPO sees residual actions on top of a frozen base.

    The wrapper does **not** subclass ``Policy``; it is a gym wrapper.
    SB3's PPO trains a policy that maps flat_obs → residual_action;
    the wrapper handles composing that residual with the frozen base
    policy's action and rewriting the env reward.

    Lifecycle per episode:

    * ``reset()`` resets the underlying env AND calls
      ``base_policy.reset()`` so the base's internal chunk buffer
      restarts from step 0. It also calls ``base.select_action(obs_0)``
      once to populate the first base action for the flat obs.
    * ``step(residual_action)`` composes the residual with the cached
      base action, steps the env, then calls ``base.select_action``
      once on the new observation to cache the next base action.
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
            base_policy: Frozen base policy.
            compositor: :class:`ResidualCompositor` instance.
            reward_fn: ``reward_fn(info, cube_xy) -> reward`` callable.
            feature_extractor: Maps the per-step obs dict to a tail
                feature vector concatenated onto the flat obs. Defaults
                to a no-op (zero-width).
            cube_state_fn: Accessor for the cube's qpos slice.
        """
        super().__init__(env)
        self._base_policy = base_policy
        self._compositor = compositor
        self._reward_fn = reward_fn
        self._feature_extractor = feature_extractor
        self._cube_state_fn = cube_state_fn

        # env.observation_space is typed gym.Space; we assume Dict-with-agent_pos
        # (the ALOHA contract). Cast through Any for mypy.
        obs_space_any: Any = env.observation_space
        agent_pos_space: Any = obs_space_any["agent_pos"]
        self._agent_pos_dim = int(agent_pos_space.shape[0])
        action_space_any: Any = env.action_space
        self._action_dim = int(action_space_any.shape[0])
        self._feature_dim = self._infer_feature_dim(env)
        flat_dim = (
            self._agent_pos_dim + _CUBE_STATE_DIM + self._action_dim + self._feature_dim
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_dim,),
            dtype=np.float32,
        )
        # action_space is inherited from the underlying env via gym.Wrapper.

        self._cached_base_action: npt.NDArray[np.float32] | None = None

    def _infer_feature_dim(self, env: gym.Env[Any, Any]) -> int:
        """Probe the feature_extractor against env's initial obs to size flat obs."""
        try:
            obs, _info = env.reset(seed=0)
            sample_features = self._feature_extractor(obs)
            return int(sample_features.shape[0])
        except Exception:  # noqa: BLE001 - graceful fallback
            return 0

    @property
    def compositor(self) -> ResidualCompositor:
        """Expose the compositor so callers (eg checkpointing) can read alpha."""
        return self._compositor

    def _build_flat_obs(
        self,
        obs_dict: Mapping[str, Any],
        base_action: npt.NDArray[np.float32],
    ) -> npt.NDArray[np.float32]:
        """Concatenate (agent_pos, cube_state, base_action, features) into a Box obs."""
        agent_pos = np.asarray(obs_dict["agent_pos"], dtype=np.float32).flatten()
        cube_state = np.asarray(
            self._cube_state_fn(self.env), dtype=np.float32
        ).flatten()
        features = np.asarray(self._feature_extractor(obs_dict), dtype=np.float32)
        return np.concatenate(
            [agent_pos, cube_state, base_action.astype(np.float32), features],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[npt.NDArray[np.float32], dict[str, Any]]:
        """Reset env + base policy and return the first flat observation."""
        obs, info = self.env.reset(seed=seed, options=options)
        self._base_policy.reset()
        # Prime the cache with the first base action so step() can compose
        # without calling base.select_action a second time.
        first_base = np.asarray(self._base_policy.select_action(obs), dtype=np.float32)
        self._cached_base_action = first_base
        return self._build_flat_obs(obs, first_base), info

    def step(
        self, action: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Compose (base + alpha * residual), step, cache next base action."""
        if self._cached_base_action is None:
            raise RuntimeError(
                "ResidualEnvWrapper.step called before reset(); call reset first."
            )
        with torch.no_grad():
            base_t = torch.from_numpy(self._cached_base_action)
            residual_t = torch.from_numpy(np.asarray(action, dtype=np.float32))
            composed_t = self._compositor(base_t, residual_t)
        composed_np = composed_t.cpu().numpy().astype(np.float32)

        obs, _native_reward, terminated, truncated, info = self.env.step(composed_np)
        cube_xy = self._cube_state_fn(self.env)[:2]
        reward = float(self._reward_fn(info, cube_xy))

        # Call base.select_action exactly once for the next step's obs.
        # ACT and other chunked policies advance their internal pointer
        # on each call, so a single call per step is non-negotiable.
        next_base = np.asarray(self._base_policy.select_action(obs), dtype=np.float32)
        self._cached_base_action = next_base
        flat_obs = self._build_flat_obs(obs, next_base)
        return flat_obs, reward, bool(terminated), bool(truncated), info

    def extract_obs_features(self, obs: Mapping[str, Any]) -> npt.NDArray[np.float32]:
        """Public wrapper around the feature_extractor for symmetry with train code."""
        return self._feature_extractor(obs)


__all__ = [
    "FeatureExtractor",
    "ResidualEnvWrapper",
    "RewardFn",
    "zero_feature_extractor",
]
