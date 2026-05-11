"""ALOHA Transfer Cube environment factory and physics-state accessor.

Wraps LeRobot's ``make_env`` for the single-env n=1 case. Also exposes the
cube's 7-element qpos slice so the geometric success detector
(:mod:`roboeval.envs.success`) can read it once per step from the underlying
``dm_control`` ``Physics`` object.

Heavy dependencies (``lerobot``, ``gym_aloha``, ``mujoco``) are imported
lazily inside :func:`make_aloha_env` so that ``import roboeval.envs.aloha``
itself stays cheap and CI-checkable without the full stack.
"""

from __future__ import annotations

from typing import Any, cast

import gymnasium as gym
import numpy as np
import numpy.typing as npt

ALOHA_TRANSFER_CUBE_ID = "gym_aloha/AlohaTransferCube-v0"
"""Canonical gym id used for logging and W&B config."""

_CUBE_QPOS_SLICE = slice(16, 23)
"""Slice into ``physics.data.qpos`` that exposes the cube pose.

The first 16 entries are the bimanual arm joints + grippers (see
``gym_aloha.constants.START_ARM_POSE`` for the layout); the last 7 entries
are the cube's free-joint state ``(x, y, z, qw, qx, qy, qz)``.
See ``gym_aloha/tasks/sim.py::TransferCubeTask.get_env_state``.
"""


def make_aloha_env(
    task: str = "AlohaTransferCube-v0",
    episode_length: int = 400,
    obs_type: str = "pixels_agent_pos",
) -> gym.Env[Any, Any]:
    """Build a single ALOHA env via LeRobot's factory, unwrap the size-1 vec env.

    LeRobot's ``make_env`` always returns a ``SyncVectorEnv``; we use
    ``n_envs=1`` and extract ``envs[0]`` so the rollout loop can use the
    simpler single-env API (no batch dim, no ``final_info`` handling).

    Args:
        task: ALOHA task id (default ``"AlohaTransferCube-v0"``).
        episode_length: ``max_episode_steps`` passed through to gym
            (PRD Section 7.2: default 400). Overrides gym-aloha's
            registered 300-step default.
        obs_type: gym-aloha observation format; ``"pixels_agent_pos"``
            matches the ACT checkpoint's input features.

    Returns:
        A single :class:`gymnasium.Env` ready for
        ``reset(seed=...)`` and ``step(action)``.
    """
    from lerobot.envs.configs import AlohaEnv as LRAlohaCfg
    from lerobot.envs.factory import make_env

    cfg = LRAlohaCfg(
        task=task,
        episode_length=episode_length,
        obs_type=obs_type,
    )
    suite = make_env(cfg, n_envs=1)
    # suite shape: {suite_name: {task_id: vec_env}}; pick the single vec env.
    vec_env = next(iter(next(iter(suite.values())).values()))
    single = vec_env.envs[0]
    return cast(gym.Env[Any, Any], single)


def get_cube_state(env: gym.Env[Any, Any]) -> npt.NDArray[np.float64]:
    """Read the cube's 7-element qpos slice from the underlying physics.

    The path walks ``env.unwrapped`` → ``gym_aloha.env.AlohaEnv`` →
    ``_env`` (the ``dm_control.rl.control.Environment``) →
    ``physics.data.qpos`` (a MuJoCo ``mjData.qpos`` view). The cube
    occupies the last 7 entries.

    Args:
        env: An ALOHA Transfer Cube env produced by :func:`make_aloha_env`.
            Must not be a ``VectorEnv`` — pass the unwrapped single env.

    Returns:
        1-D ``float64`` array of length 7: ``(x, y, z, qw, qx, qy, qz)``.
    """
    aloha = env.unwrapped
    # _env is a private dm_control attribute on gym_aloha.env.AlohaEnv;
    # mypy can't see it because the env class is dynamically loaded.
    physics = aloha._env.physics  # type: ignore[attr-defined]
    qpos = np.asarray(physics.data.qpos, dtype=np.float64)
    return qpos[_CUBE_QPOS_SLICE].copy()
