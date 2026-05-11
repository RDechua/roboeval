"""Single-episode rollout engine.

Runs one policy in one env for one episode and produces a typed
:class:`RolloutResult`. Tracks both gym-aloha's native success signal
(``info["is_success"]``, equivalent to ``reward == 4``) and the PRD's
geometric secondary criterion via
:class:`roboeval.envs.success.TransferCubeSuccessDetector`.

The cube-state accessor is parameterised (``cube_state_fn``) so unit tests
can substitute a mock without going through ``dm_control``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from roboeval.envs.aloha import get_cube_state
from roboeval.envs.success import TransferCubeSuccessDetector
from roboeval.evaluation.types import RolloutResult
from roboeval.policies.base import Policy

CubeStateFn = Callable[[gym.Env[Any, Any]], npt.NDArray[np.float64]]
"""Function that pulls the 7-element cube qpos from a live env."""


def seed_everything(seed: int) -> None:
    """Seed numpy and torch RNGs for reproducibility.

    The env itself is seeded via ``env.reset(seed=...)`` separately;
    this covers everything the policy adapter touches.

    Args:
        seed: Non-negative integer seed.
    """
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def run_rollout(
    env: gym.Env[Any, Any],
    policy: Policy,
    success_detector: TransferCubeSuccessDetector,
    seed_group: int,
    rollout_idx: int,
    episode_seed: int,
    max_steps: int = 400,
    cube_state_fn: CubeStateFn = get_cube_state,
) -> RolloutResult:
    """Run a single rollout and return a typed result.

    Args:
        env: Single (non-vectorised) ALOHA env.
        policy: Any object implementing :class:`Policy`.
        success_detector: Detector instance; the function calls
            :meth:`TransferCubeSuccessDetector.reset` itself.
        seed_group: Logical seed group this rollout belongs to (0/1/2).
        rollout_idx: Within-group rollout index.
        episode_seed: Seed passed to ``env.reset`` AND to numpy/torch.
        max_steps: Hard upper bound on env steps.
        cube_state_fn: Accessor for the cube's 7-element qpos. Overridden
            in tests; defaults to
            :func:`roboeval.envs.aloha.get_cube_state`.

    Returns:
        A :class:`RolloutResult` describing the episode.
    """
    seed_everything(episode_seed)
    policy.reset()
    success_detector.reset()

    obs, _info = env.reset(seed=episode_seed)

    success_native = False
    success_custom = False
    success_step: int | None = None
    max_reward = 0
    terminated = False
    truncated = False
    n_steps = 0
    final_cube_state = cube_state_fn(env)

    start_t = time.perf_counter()
    for step_idx in range(max_steps):
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        n_steps = step_idx + 1
        max_reward = max(max_reward, int(float(reward)))

        cube_state = cube_state_fn(env)
        final_cube_state = cube_state

        custom_hit = success_detector.update(cube_state)
        native_hit = bool(info.get("is_success", False))

        if (native_hit or custom_hit) and success_step is None:
            success_step = n_steps
        if native_hit:
            success_native = True
        if custom_hit:
            success_custom = True
        if terminated or truncated:
            break

    wall_time = time.perf_counter() - start_t
    final_z = float(final_cube_state[2])
    final_xy = float(np.hypot(final_cube_state[0], final_cube_state[1]))

    return RolloutResult(
        seed_group=seed_group,
        rollout_idx=rollout_idx,
        episode_seed=episode_seed,
        success=success_native,
        success_custom=success_custom,
        success_step=success_step,
        n_steps=n_steps,
        max_reward=max_reward,
        terminated=bool(terminated),
        truncated=bool(truncated),
        wall_time_s=wall_time,
        final_cube_z=final_z,
        final_cube_xy_dist=final_xy,
    )
