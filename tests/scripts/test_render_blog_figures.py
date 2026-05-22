"""Smoke tests for scripts.render_blog_figures."""

from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from scripts.render_blog_figures import render_cross_axis_degradation  # noqa: E402


def test_render_cross_axis_degradation_emits_png(tmp_path: Path) -> None:
    """The renderer writes a non-empty PNG to the target path."""
    repo_root = Path(__file__).resolve().parents[2]
    out_path = tmp_path / "cross_axis.png"
    render_cross_axis_degradation(
        headline_path=repo_root / "data" / "headline.json",
        out_path=out_path,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 1024  # at least 1 KB
