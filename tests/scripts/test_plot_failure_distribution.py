"""Tests for scripts/plot_failure_distribution.py pure helpers.

The matplotlib-rendering path is exercised end-to-end by writing a real
PNG to a tmp_path — small data, deterministic, fast (~0.2s).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from plot_failure_distribution import (  # noqa: E402
    build_stack_arrays,
    emit_markdown_table,
    load_distribution,
    render_figure,
)


def test_build_stack_arrays_converts_counts_to_percentages():
    cells = [
        ("nominal", {"success": 120, "timeout": 28, "needs_review": 2}, 150),
        ("+5cm", {"success": 46, "recovery_failure": 89, "needs_review": 13}, 150),
    ]
    x_labels, stacks = build_stack_arrays(cells)
    assert x_labels == ["nominal", "+5cm"]
    # Successes: 120/150 = 80.0%, 46/150 = 30.666...%
    assert stacks["success"] == pytest.approx([80.0, 30.6667], abs=1e-3)
    # Recovery present only in cell 2.
    assert stacks["recovery_failure"] == pytest.approx([0.0, 59.3333], abs=1e-3)
    # Buckets entirely absent from the input still appear in the stack at 0.
    assert stacks["grasp_failure"] == [0.0, 0.0]


def test_load_distribution_round_trips_n_rollouts(tmp_path):
    payload = {
        "schema_version": 1,
        "run_id": "test",
        "n_rollouts": 42,
        "distribution": {"success": 30, "timeout": 12},
    }
    path = tmp_path / "auto_labels_test.json"
    path.write_text(json.dumps(payload))
    dist, n = load_distribution(path)
    assert n == 42
    assert dist["success"] == 30
    assert dist["timeout"] == 12


def test_emit_markdown_table_includes_cell_and_pct():
    cells = [
        ("nominal", {"success": 120, "timeout": 28, "needs_review": 2}, 150),
    ]
    md = emit_markdown_table(cells)
    # Header is present.
    assert "| cell | n |" in md
    # Success row has both raw count and percentage.
    assert "120 (80.0%)" in md
    assert "28 (18.7%)" in md


def test_render_figure_writes_png(tmp_path):
    # matplotlib is only in the dev/runtime deps, not in CI's minimal
    # install. Skip on CI rather than fail-import; the pure-function
    # tests above still verify the data-prep path.
    pytest.importorskip("matplotlib")
    cells = [
        ("nominal", {"success": 120, "timeout": 28, "needs_review": 2}, 150),
        ("+5cm", {"success": 46, "recovery_failure": 89, "needs_review": 13}, 150),
    ]
    out_path = tmp_path / "figures" / "test.png"
    rendered = render_figure(cells, out_path)
    assert rendered == out_path
    assert out_path.exists()
    # PNG magic bytes.
    with out_path.open("rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"
