"""Tests for scripts.build_headline_json — the headline.json producer."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.build_headline_json import build_headline_payload


def test_build_headline_payload_has_eleven_cells() -> None:
    """With the real auto_labels on disk, the builder produces 10 cells:
    6 spatial (y±{1,3,5}cm) + 3 temporal (delay-{1,3,5}step) + nominal."""
    repo_root = Path(__file__).resolve().parents[2]
    payload = build_headline_payload(repo_root=repo_root)
    assert payload["schema_version"] == 1
    cells = payload["cells"]
    assert len(cells) == 10
    axes = {c["axis"] for c in cells}
    assert axes == {"spatial", "temporal", "nominal"}


def test_build_headline_payload_failure_counts_sum_to_n_rollouts() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    payload = build_headline_payload(repo_root=repo_root)
    for cell in payload["cells"]:
        counts = cell["failure_counts"]
        assert sum(counts.values()) == cell["n_rollouts"]


def test_build_headline_payload_spatial_cells_have_known_means() -> None:
    """Cross-check a few cells against the STATE.md headline table."""
    repo_root = Path(__file__).resolve().parents[2]
    payload = build_headline_payload(repo_root=repo_root)
    by_id = {c["cell_id"]: c for c in payload["cells"]}
    assert by_id["y-5cm"]["mean_tsr_custom"] == 0.127
    assert by_id["y+5cm"]["mean_tsr_custom"] == 0.307
    assert by_id["delay-5step"]["mean_tsr_custom"] == 0.687
    assert by_id["nominal"]["mean_tsr_custom"] == 0.800


def test_headline_json_file_committed_and_valid() -> None:
    """The repository must contain a tracked data/headline.json
    matching the schema produced by the build script."""
    repo_root = Path(__file__).resolve().parents[2]
    headline = json.loads((repo_root / "data" / "headline.json").read_text())
    assert headline["schema_version"] == 1
    assert len(headline["cells"]) == 10
