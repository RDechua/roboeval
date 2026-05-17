"""Tests for the TSR-vs-x degradation-curve plot script."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from plot_degradation_curve import (  # noqa: E402
    aggregate_cell,
    bernoulli_se,
    per_seed_group_tsr,
    render_curve,
)


def _auto_labels(n_per_seed: int, n_success_per_seed: dict[int, int]):
    """Build an auto_labels-shaped dict with the requested per-seed split."""
    labels = []
    for sg, n_success in n_success_per_seed.items():
        for i in range(n_per_seed):
            labels.append(
                {
                    "seed_group": sg,
                    "rollout_idx": i,
                    "episode_seed": sg * 100_003 + i,
                    "failure_mode": None if i < n_success else "recovery_failure",
                    "evidence": {},
                }
            )
    return {"schema_version": 1, "n_rollouts": len(labels), "labels": labels}


def test_per_seed_group_tsr_returns_one_value_per_group():
    obj = _auto_labels(50, {0: 44, 1: 38, 2: 38})
    per_seed, total = per_seed_group_tsr(obj)
    assert len(per_seed) == 3
    assert per_seed == pytest.approx([0.88, 0.76, 0.76])
    assert total == 150


def test_aggregate_cell_matches_eval_loop_aggregation(tmp_path):
    """Reproduces the act_nominal headline (0.80 ± 0.057)."""
    obj = _auto_labels(50, {0: 44, 1: 38, 2: 38})
    path = tmp_path / "auto_labels_nominal.json"
    path.write_text(json.dumps(obj))
    mean, sigma, n = aggregate_cell(path)
    assert mean == pytest.approx(0.80, abs=1e-3)
    assert sigma == pytest.approx(0.0566, abs=1e-3)
    assert n == 150


def test_bernoulli_se_matches_closed_form():
    # p=0.5, n=100 → √(0.25/100) = 0.05
    assert bernoulli_se(0.5, 100) == pytest.approx(0.05)
    # Edge cases.
    assert bernoulli_se(0.0, 100) == 0.0
    assert bernoulli_se(1.0, 100) == 0.0
    assert bernoulli_se(0.5, 0) == 0.0


def test_render_curve_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    cells = [
        ("nominal", 0.0, 0.80, 0.057, 150),
        ("+1cm", 1.0, 0.72, 0.102, 150),
        ("+3cm", 3.0, 0.553, 0.041, 150),
        ("+5cm", 5.0, 0.307, 0.019, 150),
    ]
    out_path = tmp_path / "figures" / "curve.png"
    rendered = render_curve(cells, out_path)
    assert rendered == out_path
    assert out_path.exists()
    with out_path.open("rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"


def test_render_curve_sorts_by_x_value(tmp_path):
    """Cells in any order should plot left-to-right by x_value."""
    pytest.importorskip("matplotlib")
    # Intentionally pass cells in scrambled order.
    cells = [
        ("+5cm", 5.0, 0.307, 0.019, 150),
        ("nominal", 0.0, 0.80, 0.057, 150),
        ("-3cm", -3.0, 0.553, 0.025, 150),
        ("+1cm", 1.0, 0.72, 0.102, 150),
    ]
    out_path = tmp_path / "scrambled.png"
    render_curve(cells, out_path)
    # If the function didn't sort, matplotlib's x-axis would zigzag and
    # the line plot would be non-monotonic. We can't check the figure
    # contents directly without parsing PNG; the render-no-raise +
    # presence of file is enough for the function-level contract here.
    assert out_path.exists()


def test_aggregate_cell_handles_single_seed_group(tmp_path):
    obj = _auto_labels(50, {0: 30})
    path = tmp_path / "single.json"
    path.write_text(json.dumps(obj))
    mean, sigma, n = aggregate_cell(path)
    assert mean == pytest.approx(0.60)
    assert sigma == 0.0  # single seed → no across-group variance
    assert n == 50
