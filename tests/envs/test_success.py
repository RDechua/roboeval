"""Unit tests for TransferCubeSuccessDetector.

These tests use synthetic 7-element cube-state arrays and assert that the
detector's dwell counter and success signal evolve correctly. No env, no
torch.
"""

from __future__ import annotations

import numpy as np
import pytest

from roboeval.envs.success import SuccessCriterion, TransferCubeSuccessDetector

# SuccessCriterion has no defaults (calibration values come from the JSON;
# PRD constants from configs). These per-test defaults reproduce the old
# Week-2 placeholder behaviour so the dwell-counter tests stay readable.
_TEST_DEFAULTS = {
    "z_threshold_m": 0.05,
    "xy_tolerance_m": 0.05,
    "dwell_steps": 5,
    "target_xy": (0.0, 0.0),
}


def _crit(**overrides):
    return SuccessCriterion(**{**_TEST_DEFAULTS, **overrides})


def _cube_state(z, xy=(0.0, 0.0)):
    """Synthetic cube qpos for tests."""
    return np.array([xy[0], xy[1], z, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def test_below_threshold_never_succeeds():
    det = TransferCubeSuccessDetector(_crit())
    for _ in range(20):
        assert det.update(_cube_state(z=0.01)) is False
    assert det.dwell_counter == 0


def test_dwell_fires_at_exact_step():
    crit = _crit(z_threshold_m=0.05, dwell_steps=5)
    det = TransferCubeSuccessDetector(crit)
    results = [det.update(_cube_state(z=0.1)) for _ in range(6)]
    assert results == [False, False, False, False, True, True]


def test_dwell_resets_on_miss():
    crit = _crit(z_threshold_m=0.05, dwell_steps=5)
    det = TransferCubeSuccessDetector(crit)
    for _ in range(3):
        det.update(_cube_state(z=0.1))
    assert det.dwell_counter == 3
    det.update(_cube_state(z=0.01))
    assert det.dwell_counter == 0
    results = [det.update(_cube_state(z=0.1)) for _ in range(5)]
    assert results == [False, False, False, False, True]


def test_xy_outside_box_fails():
    crit = _crit(xy_tolerance_m=0.05, target_xy=(0.0, 0.0))
    det = TransferCubeSuccessDetector(crit)
    assert det.update(_cube_state(z=0.1, xy=(0.10, 0.10))) is False
    assert det.dwell_counter == 0


def test_xy_inside_box_succeeds():
    crit = _crit(xy_tolerance_m=0.05, dwell_steps=2)
    det = TransferCubeSuccessDetector(crit)
    det.update(_cube_state(z=0.1, xy=(0.03, 0.03)))
    assert det.update(_cube_state(z=0.1, xy=(0.03, 0.03))) is True


def test_target_xy_offset():
    crit = _crit(xy_tolerance_m=0.05, dwell_steps=1, target_xy=(0.5, 0.0))
    det = TransferCubeSuccessDetector(crit)
    # cube near target_xy=(0.5, 0.0) inside box
    assert det.update(_cube_state(z=0.1, xy=(0.49, 0.01))) is True
    det.reset()
    # cube at origin, far from target_xy=(0.5, 0.0)
    assert det.update(_cube_state(z=0.1, xy=(0.0, 0.0))) is False


def test_invalid_shape_raises():
    det = TransferCubeSuccessDetector(_crit())
    with pytest.raises(ValueError, match=r"cube_state must have shape"):
        det.update(np.zeros(3))


def test_reset_clears_counter():
    crit = _crit(dwell_steps=3)
    det = TransferCubeSuccessDetector(crit)
    for _ in range(2):
        det.update(_cube_state(z=0.1))
    assert det.dwell_counter == 2
    det.reset()
    assert det.dwell_counter == 0
