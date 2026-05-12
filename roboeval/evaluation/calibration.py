"""Data-driven calibration of the geometric success criterion.

PRD Section 6.2 defines a secondary TSR (``mean_tsr_custom``) based on
the cube ending up in a target xy-box of half-width ``xy_tolerance_m``
around ``target_xy``, sustained for ``dwell_steps`` simulation steps.
The defaults shipped in :mod:`roboeval.envs.success` (``target_xy=(0,0)``,
``xy_tolerance_m=0.05``) are placeholders; the actual cube transfer
endpoint depends on the ALOHA receptacle geometry and the policy.

This module derives those constants empirically from N nominal rollouts:

1. Run :func:`evaluate_policy` once and collect a list of ``RolloutResult``.
2. Filter to primary-successful rollouts (``result.success``).
3. ``target_xy`` = centroid of ``(final_cube_x, final_cube_y)`` over
   successes.
4. ``xy_tolerance_m`` = 90th percentile of
   ``|| (x_i, y_i) - target_xy ||`` across successes.

The 90th percentile (not 100th) trades a 10pp coverage loss for
robustness against the long tail of barely-succeeded rollouts whose
cube ends up at the edge of the receptacle.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from roboeval.evaluation.types import RolloutResult

_MIN_SUCCESSES_REQUIRED: int = 25
"""Refuse to derive a centroid from fewer than this many successes.

At N=50 and p≈0.8, expected successes ≈ 40. A run that produces <25
successes likely has an upstream problem (wrong checkpoint, MPS
fallback) and writing a calibration JSON from it would silently encode
that pathology. Fail loudly instead.
"""

_PERCENTILE: float = 90.0
"""Percentile used for ``xy_tolerance_m``. See module docstring."""


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Frozen outcome of one calibration run.

    Attributes:
        target_xy: Calibrated target xy as ``(x_mean, y_mean)`` in metres.
        xy_tolerance_m: Calibrated tolerance: the 90th percentile of
            successful-endpoint distances from ``target_xy``.
        percentile: The percentile used (always :data:`_PERCENTILE`).
        n_rollouts: Total rollouts that fed the calibration.
        n_successes: Rollouts with primary ``success=True`` whose
            endpoints were used to compute the centroid + tolerance.
        success_endpoints: Per-success ``(x, y)`` tuples, in input order.
            Stored so the calibration can be re-derived offline without
            re-running the policy.
    """

    target_xy: tuple[float, float]
    xy_tolerance_m: float
    percentile: float
    n_rollouts: int
    n_successes: int
    success_endpoints: tuple[tuple[float, float], ...]


def calibrate_target_xy(rollouts: Sequence[RolloutResult]) -> CalibrationResult:
    """Compute ``target_xy`` and ``xy_tolerance_m`` from rollout endpoints.

    Args:
        rollouts: Per-rollout results from a nominal evaluation run.
            Must contain at least :data:`_MIN_SUCCESSES_REQUIRED`
            primary-successful entries.

    Returns:
        A frozen :class:`CalibrationResult` ready to write to the
        calibration JSON sidecar.

    Raises:
        ValueError: If ``rollouts`` is empty, or the number of primary
            successes is below :data:`_MIN_SUCCESSES_REQUIRED`. The
            error message names the count so the operator can either
            re-run or bump ``--n-rollouts``.
    """
    if not rollouts:
        raise ValueError("calibrate_target_xy requires at least one rollout")

    successes = [r for r in rollouts if r.success]
    n_successes = len(successes)
    if n_successes < _MIN_SUCCESSES_REQUIRED:
        raise ValueError(
            f"calibrate_target_xy requires >= {_MIN_SUCCESSES_REQUIRED} "
            f"primary successes; got {n_successes} from "
            f"{len(rollouts)} rollouts. Either re-run with a different "
            "seed or increase --n-rollouts."
        )

    xs = [r.final_cube_x for r in successes]
    ys = [r.final_cube_y for r in successes]
    target_x = statistics.fmean(xs)
    target_y = statistics.fmean(ys)

    # Euclidean distance from centroid for each successful endpoint
    distances = [
        ((x - target_x) ** 2 + (y - target_y) ** 2) ** 0.5
        for x, y in zip(xs, ys, strict=True)
    ]
    distances_sorted = sorted(distances)
    # Linear interpolation between order statistics for percentile estimation
    rank = (_PERCENTILE / 100.0) * (len(distances_sorted) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(distances_sorted) - 1)
    frac = rank - lo
    xy_tolerance = distances_sorted[lo] * (1 - frac) + distances_sorted[hi] * frac

    endpoints = tuple((float(x), float(y)) for x, y in zip(xs, ys, strict=True))
    return CalibrationResult(
        target_xy=(target_x, target_y),
        xy_tolerance_m=float(xy_tolerance),
        percentile=_PERCENTILE,
        n_rollouts=len(rollouts),
        n_successes=n_successes,
        success_endpoints=endpoints,
    )


def calibration_to_dict(
    result: CalibrationResult,
    *,
    git_sha: str,
    timestamp: str,
    source_config: str,
    policy_id: str,
    env_id: str,
) -> dict[str, Any]:
    """Render a :class:`CalibrationResult` as a JSON-serialisable dict.

    Args:
        result: Calibration outcome.
        git_sha: Git SHA of the commit that produced this calibration;
            for downstream provenance.
        timestamp: ISO-8601 UTC timestamp string.
        source_config: Path of the YAML config used for the calibration
            rollouts.
        policy_id: HuggingFace repo id of the policy.
        env_id: Gymnasium env id.

    Returns:
        A plain dict suitable for ``json.dump``.
    """
    return {
        "target_xy": list(result.target_xy),
        "xy_tolerance_m": result.xy_tolerance_m,
        "percentile": result.percentile,
        "n_rollouts": result.n_rollouts,
        "n_successes": result.n_successes,
        "git_sha": git_sha,
        "timestamp": timestamp,
        "source_config": source_config,
        "policy_id": policy_id,
        "env_id": env_id,
        "success_endpoints": [list(p) for p in result.success_endpoints],
    }
