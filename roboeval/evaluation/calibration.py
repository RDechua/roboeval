"""Data-driven calibration of the geometric success criterion.

PRD Section 6.2 defines a secondary TSR (``mean_tsr_custom``) based on
the cube ending up in a target xy-box of half-width ``xy_tolerance_m``
around ``target_xy``, sustained for ``dwell_steps`` simulation steps.
``z_threshold_m`` and ``dwell_steps`` are PRD constants; ``target_xy``
and ``xy_tolerance_m`` are calibration-derived and live in
``data/calibration/transfer_cube_target_xy.json``. Eval configs
interpolate the calibrated values via the ``${calibration:...}``
OmegaConf resolver registered by the CLI; see
:func:`register_calibration_resolver`.

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

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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


DEFAULT_CALIBRATION_PATH = "data/calibration/transfer_cube_target_xy.json"
"""Project-standard path for the frozen Transfer-Cube calibration JSON."""

_REQUIRED_CALIBRATION_KEYS: frozenset[str] = frozenset({"target_xy", "xy_tolerance_m"})
"""Keys a calibration JSON must contain to be usable in eval configs."""

_CALIBRATION_CACHE: dict[str, dict[str, Any]] = {}
"""Per-path cache so the JSON is read once per process, not per resolver call."""


def load_calibration_json(
    path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> dict[str, Any]:
    """Load and validate a frozen calibration JSON.

    Args:
        path: Path to the calibration JSON. Defaults to
            :data:`DEFAULT_CALIBRATION_PATH`.

    Returns:
        The parsed JSON as a dict. Guaranteed to contain at least the
        keys in :data:`_REQUIRED_CALIBRATION_KEYS`.

    Raises:
        FileNotFoundError: If the calibration JSON is missing. The error
            message points the operator at ``roboeval calibrate``.
        KeyError: If the JSON is missing one of the required keys.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Calibration JSON not found at {p}. "
            f"Run `roboeval calibrate --config configs/baseline/act_nominal_fast.yaml "
            f"--output {p}` to generate it."
        )
    data: dict[str, Any] = json.loads(p.read_text())
    missing = _REQUIRED_CALIBRATION_KEYS - set(data.keys())
    if missing:
        raise KeyError(
            f"Calibration JSON {p} is missing required keys: {sorted(missing)}. "
            f"Re-run `roboeval calibrate` to regenerate."
        )
    return data


def clear_calibration_cache() -> None:
    """Drop the in-process calibration cache.

    Useful in tests that write multiple calibration JSONs to disk and
    need the resolver to re-read between them.
    """
    _CALIBRATION_CACHE.clear()


def register_calibration_resolver(
    path: str | Path = DEFAULT_CALIBRATION_PATH,
    name: str = "calibration",
    replace: bool = True,
) -> None:
    """Register an OmegaConf resolver that pulls values from the calibration JSON.

    Once registered, YAML configs can interpolate calibrated values
    directly::

        success:
          target_xy:      ${calibration:target_xy}
          xy_tolerance_m: ${calibration:xy_tolerance_m}

    The JSON is read on first interpolation and cached for the rest of
    the process; call :func:`clear_calibration_cache` to force a re-read.

    Args:
        path: Path to the calibration JSON; passed to
            :func:`load_calibration_json` lazily on first interpolation.
        name: Resolver name as it appears in YAML
            (``${<name>:<key>}``). Default ``"calibration"``.
        replace: When ``True`` (default), re-registering replaces any
            existing resolver of the same name. Set ``False`` to make
            re-registration a no-op.

    Raises:
        ValueError: If ``path`` resolves to an empty string after
            casting (would silently default to cwd).
    """
    from omegaconf import OmegaConf

    resolved_path = str(Path(path))
    if not resolved_path:
        raise ValueError("calibration JSON path must be non-empty")

    def _resolve(key: str) -> Any:
        if resolved_path not in _CALIBRATION_CACHE:
            _CALIBRATION_CACHE[resolved_path] = load_calibration_json(resolved_path)
        data = _CALIBRATION_CACHE[resolved_path]
        if key not in data:
            raise KeyError(
                f"Calibration key '{key}' not found in {resolved_path}. "
                f"Available keys: {sorted(data.keys())}"
            )
        return data[key]

    OmegaConf.register_new_resolver(name, _resolve, replace=replace)
