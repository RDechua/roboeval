"""Tests for the dashboard data loaders.

These read only tracked artifacts (``data/headline.json`` schema v2
and ``docs/figures/phase4_ablation.json``) plus synthetic tmp_path
fixtures — no dependency on the gitignored ``outputs/`` or
``data/taxonomy/`` files, so they pass in CI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboeval.dashboard.data import (
    load_all,
    load_dashboard_data,
    load_headline_json,
    load_phase4_ablation,
)
from roboeval.dashboard.models import DashboardData


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_load_headline_json_returns_ten_cells() -> None:
    cells = load_headline_json(_repo_root() / "data" / "headline.json")
    assert len(cells) == 10


def test_load_headline_json_rejects_wrong_schema_version(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 99, "cells": []}))
    with pytest.raises(ValueError, match="schema_version"):
        load_headline_json(bad)


def test_load_phase4_ablation_three_conditions() -> None:
    conds, welches = load_phase4_ablation(
        _repo_root() / "docs" / "figures" / "phase4_ablation.json"
    )
    assert {c.condition_id for c in conds} == {"A", "B", "C"}
    assert {w.arm_id for w in welches} == {"B", "C"}


def test_load_dashboard_data_populates_ablation_failure_counts() -> None:
    """Schema v2 carries ablation + failure_counts inline.

    Validates the canonical Phase 4 numbers (Condition B sparse:
    106 recovery_failure rollouts) without any gitignored deps.
    """
    data = load_dashboard_data(_repo_root() / "data" / "headline.json")
    by_id = {c.condition_id: c for c in data.ablation}
    assert by_id["B"].failure_counts.recovery_failure == 106


def test_load_dashboard_data_rejects_non_v2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": 1, "cells": []}))
    with pytest.raises(ValueError, match="schema_version 2"):
        load_dashboard_data(bad)


def test_load_all_returns_dashboard_data() -> None:
    data = load_all(repo_root=_repo_root())
    assert isinstance(data, DashboardData)
    assert len(data.cells) == 10
    assert len(data.ablation) == 3
    assert len(data.welch_tests) == 2
