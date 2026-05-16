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

CUBE_GEOM_NAME = "red_box"
"""Geom name for the cube in the Transfer Cube task MJCF."""

LEFT_GRIPPER_BODY = "vx300s_left/gripper_link"
RIGHT_GRIPPER_BODY = "vx300s_right/gripper_link"
"""Body names whose ``xpos`` is used as the per-arm end-effector position."""


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


def get_gripper_xy(
    env: gym.Env[Any, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]] | None:
    """Read the xy position of both gripper-link bodies.

    Used by :func:`roboeval.evaluation.rollout.run_rollout` to compute the
    terminal-step EE-to-cube distance for the failure-mode classifier
    (PRD §7.2 Approach Failure rule).

    Args:
        env: An ALOHA env from :func:`make_aloha_env`.

    Returns:
        A ``(left_xy, right_xy)`` tuple of length-2 float64 arrays, or
        ``None`` if the env doesn't expose dm_control physics (mock envs
        in unit tests).
    """
    try:
        aloha = env.unwrapped
        physics = aloha._env.physics  # type: ignore[attr-defined]
        named_xpos = physics.named.data.xpos
        left = np.asarray(named_xpos[LEFT_GRIPPER_BODY], dtype=np.float64)[:2].copy()
        right = np.asarray(named_xpos[RIGHT_GRIPPER_BODY], dtype=np.float64)[:2].copy()
        return left, right
    except (AttributeError, KeyError):
        return None


def get_cube_gripper_contact(env: gym.Env[Any, Any]) -> bool:
    """Detect whether the cube is in contact with any gripper-finger geom.

    Iterates ``physics.data.contact`` for the current step; returns ``True``
    if any contact pair matches ``(red_box, *gripper_finger*)`` in either
    order. Mirrors gym-aloha's own grasp-detection logic
    (``gym_aloha/tasks/sim.py::TransferCubeTask.get_reward``) but is more
    permissive: matches *all four* finger geoms (left + right finger of
    each arm) instead of one per arm.

    Args:
        env: An ALOHA env from :func:`make_aloha_env`.

    Returns:
        ``True`` if cube↔finger contact is active this step; ``False``
        otherwise (including when the env doesn't expose dm_control physics).
    """
    try:
        aloha = env.unwrapped
        physics = aloha._env.physics  # type: ignore[attr-defined]
        for i_contact in range(int(physics.data.ncon)):
            geom1 = int(physics.data.contact[i_contact].geom1)
            geom2 = int(physics.data.contact[i_contact].geom2)
            name1 = physics.model.id2name(geom1, "geom")
            name2 = physics.model.id2name(geom2, "geom")
            if name1 == CUBE_GEOM_NAME and name2 and "gripper_finger" in name2:
                return True
            if name2 == CUBE_GEOM_NAME and name1 and "gripper_finger" in name1:
                return True
        return False
    except AttributeError:
        return False
