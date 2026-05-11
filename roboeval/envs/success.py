"""Geometric success detector for the ALOHA Transfer Cube task.

Implements the PRD Section 6.2 secondary success criterion: the cube's
centre-of-mass z-position is above ``z_threshold_m`` AND its xy-position
lies within ``xy_tolerance_m`` of the target receptacle centre, sustained
for ``dwell_steps`` consecutive simulation steps. The signal is reported
alongside gym-aloha's native contact-based primary signal (PRD: dual TSR
reporting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_CUBE_STATE_DIM = 7  # x, y, z, qw, qx, qy, qz


@dataclass(frozen=True, slots=True)
class SuccessCriterion:
    """Thresholds and target for the geometric success criterion.

    Attributes:
        z_threshold_m: Minimum cube z-position above the table, in metres.
            PRD default 0.05 m.
        xy_tolerance_m: Half-width of the target xy-box around
            ``target_xy``, in metres. PRD default 0.05 m.
        dwell_steps: Number of consecutive simulation steps the geometric
            condition must hold. PRD default 5.
        target_xy: (x, y) coordinates of the target receptacle centre,
            in metres. Default ``(0.0, 0.0)``; Week 2 tunes this against
            the ACT nominal run.
    """

    z_threshold_m: float = 0.05
    xy_tolerance_m: float = 0.05
    dwell_steps: int = 5
    target_xy: tuple[float, float] = (0.0, 0.0)


class TransferCubeSuccessDetector:
    """Stateful per-rollout detector — call :meth:`update` once per env step.

    The detector tracks a dwell counter that increments on each step the
    geometric criterion holds and resets to zero otherwise. Once the
    counter reaches ``criterion.dwell_steps``, :meth:`update` returns
    ``True`` and continues returning ``True`` as long as the criterion
    keeps holding. The caller is responsible for any latching behaviour;
    the detector itself does not latch.
    """

    def __init__(self, criterion: SuccessCriterion) -> None:
        """Construct a detector.

        Args:
            criterion: Thresholds and target coordinates.
        """
        self._criterion = criterion
        self._dwell_counter = 0

    def reset(self) -> None:
        """Clear the dwell counter for a new rollout."""
        self._dwell_counter = 0

    @property
    def dwell_counter(self) -> int:
        """Current dwell counter (read-only — for tests/diagnostics)."""
        return self._dwell_counter

    @property
    def criterion(self) -> SuccessCriterion:
        """The frozen criterion this detector was constructed with."""
        return self._criterion

    def update(self, cube_state: npt.NDArray[np.floating]) -> bool:
        """Feed one simulation step's cube state and return the success flag.

        Args:
            cube_state: 7-element ``(x, y, z, qw, qx, qy, qz)`` cube pose.
                Typically obtained from
                ``roboeval.envs.aloha.get_cube_state(env)``.

        Returns:
            ``True`` iff the geometric criterion has held for at least
            ``dwell_steps`` consecutive calls (this one inclusive).

        Raises:
            ValueError: If ``cube_state`` is not a 7-element array.
        """
        if cube_state.shape != (_CUBE_STATE_DIM,):
            raise ValueError(
                f"cube_state must have shape ({_CUBE_STATE_DIM},); "
                f"got {cube_state.shape}"
            )
        x = float(cube_state[0])
        y = float(cube_state[1])
        z = float(cube_state[2])
        xy_dist = math.hypot(
            x - self._criterion.target_xy[0],
            y - self._criterion.target_xy[1],
        )
        in_zone = (
            z > self._criterion.z_threshold_m
            and xy_dist <= self._criterion.xy_tolerance_m
        )
        if in_zone:
            self._dwell_counter += 1
        else:
            self._dwell_counter = 0
        return self._dwell_counter >= self._criterion.dwell_steps
