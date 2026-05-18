"""Reward functions for the residual PPO training loop (PRD §8.2).

Two components, ablated separately (PRD §8.3 conditions B and C):

* :func:`sparse_success_reward` — ``+1`` on ``info["is_success"]``,
  ``0.0`` otherwise. Matches gym-aloha's native success signal.
* :func:`shaped_distance_reward` — negative L2 distance of the cube's
  xy position from the calibrated target (``data/calibration/
  transfer_cube_target_xy.json``). Encourages the residual to drive the
  cube toward the receptacle even when the sparse signal is silent.

:func:`combined_reward` is the convex sum used at runtime; the wrapper
passes the per-step ``info`` dict + cube xy + target + shaping weight.
A weight of ``0.0`` is the sparse-only Condition B; any positive weight
is Condition C (shaped).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt


def sparse_success_reward(info: Mapping[str, Any]) -> float:
    """Return ``1.0`` if ``info["is_success"]`` else ``0.0``.

    Args:
        info: The per-step info dict returned by ``env.step``.

    Returns:
        ``1.0`` on success, ``0.0`` otherwise. Robust to missing key.
    """
    return 1.0 if bool(info.get("is_success", False)) else 0.0


def shaped_distance_reward(
    cube_xy: npt.NDArray[np.float64],
    target_xy: tuple[float, float],
) -> float:
    """Negative L2 distance between cube xy and target xy (metres).

    Always non-positive; closer to zero = closer to target. Use small
    weights when combining with the sparse signal to avoid drowning out
    the binary success bonus.

    Args:
        cube_xy: Length-2 numpy array of cube position in metres.
        target_xy: Calibrated transfer target (typically from
            ``data/calibration/transfer_cube_target_xy.json``).

    Returns:
        ``-sqrt((cx-tx)^2 + (cy-ty)^2)`` as a Python float.
    """
    dx = float(cube_xy[0]) - target_xy[0]
    dy = float(cube_xy[1]) - target_xy[1]
    return -float(np.hypot(dx, dy))


def combined_reward(
    info: Mapping[str, Any],
    cube_xy: npt.NDArray[np.float64],
    target_xy: tuple[float, float],
    *,
    shaping_weight: float = 0.0,
) -> float:
    """Sum sparse success + ``shaping_weight * shaped distance``.

    PRD §8.3:

    * Condition B (sparse-only): ``shaping_weight=0.0``.
    * Condition C (shaped):      ``shaping_weight>0.0``.

    Args:
        info: ``env.step`` info dict.
        cube_xy: Length-2 cube position in metres.
        target_xy: Calibrated transfer target.
        shaping_weight: Multiplier on the distance term. ``0.0`` for
            sparse-only training; ``0.01`` is a reasonable starting
            point for shaped training (subject to ablation tuning).

    Returns:
        Combined reward as a Python float.
    """
    sparse = sparse_success_reward(info)
    if shaping_weight == 0.0:
        return sparse
    return sparse + shaping_weight * shaped_distance_reward(cube_xy, target_xy)


__all__ = ["combined_reward", "shaped_distance_reward", "sparse_success_reward"]
