"""Smoke tests for the 6 spatial-perturbation cell configs.

Each cell inherits the nominal eval block via the ``extends:`` key. The
parent uses ``${calibration:target_xy}`` so we register the calibration
resolver against the frozen artifact before loading. The tests check
that:

* Every cell parses and resolves end-to-end (the eval CLI would not
  crash on it).
* The perturbation block carries the expected ``dy_m`` sign+magnitude.
* The inherited blocks (policy, env, eval, success) are present after
  the merge — guards against an over-eager `extends:` rewrite
  accidentally dropping parent fields.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from roboeval.evaluation.calibration import (
    clear_calibration_cache,
    register_calibration_resolver,
)
from roboeval.evaluation.config import load_eval_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CALIB_PATH = _REPO_ROOT / "data" / "calibration" / "transfer_cube_target_xy.json"
_SPATIAL_DIR = _REPO_ROOT / "configs" / "perturbation" / "spatial"


_SPATIAL_CELLS = [
    ("act_spatial_y-5cm.yaml", -0.05),
    ("act_spatial_y-3cm.yaml", -0.03),
    ("act_spatial_y-1cm.yaml", -0.01),
    ("act_spatial_y+1cm.yaml", 0.01),
    ("act_spatial_y+3cm.yaml", 0.03),
    ("act_spatial_y+5cm.yaml", 0.05),
]


@pytest.fixture(autouse=True)
def _register_calibration_for_test():
    """Re-register the resolver against the frozen calibration artifact."""
    clear_calibration_cache()
    register_calibration_resolver(path=_CALIB_PATH)
    yield
    clear_calibration_cache()


@pytest.mark.parametrize(("filename", "expected_dy_m"), _SPATIAL_CELLS)
def test_spatial_cell_loads_and_resolves(filename: str, expected_dy_m: float):
    cfg = load_eval_config(_SPATIAL_DIR / filename)
    # Perturbation block matches the cell's filename intent.
    assert cfg.perturbation.kind == "spatial"
    assert cfg.perturbation.dx_m == 0.0
    assert cfg.perturbation.dy_m == pytest.approx(expected_dy_m)
    # Inherited blocks survived the merge.
    assert cfg.policy.kind == "act"
    assert cfg.env.task == "AlohaTransferCube-v0"
    assert int(cfg.eval.n_rollouts_per_seed) == 50
    assert len(list(cfg.eval.seeds)) == 3
    # Calibration interpolation resolved (not a string anymore).
    assert isinstance(cfg.success.xy_tolerance_m, float)
    assert len(list(cfg.success.target_xy)) == 2


def test_spatial_suite_spans_minus_5_to_plus_5_symmetrically():
    """The 6-cell suite covers -5, -3, -1, +1, +3, +5 cm without gaps."""
    dy_values = sorted(dy for _, dy in _SPATIAL_CELLS)
    assert dy_values == pytest.approx([-0.05, -0.03, -0.01, 0.01, 0.03, 0.05])


def test_negative_and_positive_cells_have_distinct_wandb_prefixes():
    """Run dashboards must not collide across cells."""
    prefixes: set[str] = set()
    for filename, _ in _SPATIAL_CELLS:
        cfg = load_eval_config(_SPATIAL_DIR / filename)
        prefixes.add(str(cfg.wandb.name_prefix))
    assert len(prefixes) == len(_SPATIAL_CELLS)
