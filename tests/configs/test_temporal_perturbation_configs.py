"""Smoke tests for the 3 temporal-delay perturbation cell configs.

Mirrors ``test_spatial_perturbation_configs.py``: every cell parses and
resolves end-to-end, the perturbation block carries the expected
``delay_steps``, inherited parent blocks survive the merge, and W&B
name prefixes are unique.
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
_TEMPORAL_DIR = _REPO_ROOT / "configs" / "perturbation" / "temporal"


_TEMPORAL_CELLS = [
    ("act_temporal_delay_1step.yaml", 1),
    ("act_temporal_delay_3steps.yaml", 3),
    ("act_temporal_delay_5steps.yaml", 5),
]


@pytest.fixture(autouse=True)
def _register_calibration_for_test():
    clear_calibration_cache()
    register_calibration_resolver(path=_CALIB_PATH)
    yield
    clear_calibration_cache()


@pytest.mark.parametrize(("filename", "expected_delay"), _TEMPORAL_CELLS)
def test_temporal_cell_loads_and_resolves(filename: str, expected_delay: int):
    cfg = load_eval_config(_TEMPORAL_DIR / filename)
    assert cfg.perturbation.kind == "temporal"
    assert int(cfg.perturbation.delay_steps) == expected_delay
    # Inherited blocks survived the merge.
    assert cfg.policy.kind == "act"
    assert cfg.env.task == "AlohaTransferCube-v0"
    assert int(cfg.eval.n_rollouts_per_seed) == 50
    assert len(list(cfg.eval.seeds)) == 3


def test_temporal_suite_covers_three_intensities():
    delays = sorted(d for _, d in _TEMPORAL_CELLS)
    assert delays == [1, 3, 5]


def test_temporal_cells_have_distinct_wandb_prefixes():
    prefixes: set[str] = set()
    for filename, _ in _TEMPORAL_CELLS:
        cfg = load_eval_config(_TEMPORAL_DIR / filename)
        prefixes.add(str(cfg.wandb.name_prefix))
    assert len(prefixes) == len(_TEMPORAL_CELLS)
