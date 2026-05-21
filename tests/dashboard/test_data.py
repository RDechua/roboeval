"""Tests for the dashboard data loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboeval.dashboard.data import (
    load_all,
    load_headline_json,
    load_phase4_ablation,
    load_phase4_eval_results,
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


def test_load_phase4_eval_results_populates_failure_counts() -> None:
    counts_by_condition = load_phase4_eval_results(
        a_path=_repo_root()
        / "outputs"
        / "eval"
        / "act_spatial_y+5cm"
        / "eval_results_w6k2wole.json",
        b_path=_repo_root()
        / "outputs"
        / "residual"
        / "y+5cm_sparse"
        / "eval_results_o6ukyo53.json",
        c_path=_repo_root()
        / "outputs"
        / "residual"
        / "y+5cm_shaped"
        / "eval_results_43czuigy.json",
    )
    assert set(counts_by_condition.keys()) == {"A", "B", "C"}
    assert counts_by_condition["B"].recovery_failure == 106


def test_load_all_returns_dashboard_data() -> None:
    data = load_all(repo_root=_repo_root())
    assert isinstance(data, DashboardData)
    assert len(data.cells) == 10
    assert len(data.ablation) == 3
    assert len(data.welch_tests) == 2
