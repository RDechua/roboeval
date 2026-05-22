"""Tests for scripts.relabel_score — Cohen's κ scoring of relabel samples."""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest

from scripts.relabel_score import score_one


def _write_sample(
    path: Path,
    *,
    run_id: str,
    unlock_at: str,
    rollouts: list[tuple[int, int, str]],
) -> None:
    """Write a relabel_sample_<run_id>.json with the given rollouts."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "auto_labels_source": f"data/taxonomy/auto_labels_{run_id}.json",
                "auto_labels_sha256": "deadbeef",
                "run_id": run_id,
                "per_category_n": 5,
                "exported_at": "2026-05-03T00:00:00+00:00",
                "unlock_at": unlock_at,
                "samples": [
                    {
                        "seed_group": sg,
                        "rollout_idx": ri,
                        "episode_seed": ri,
                        "manual_failure_mode": label,
                    }
                    for sg, ri, label in rollouts
                ],
            }
        )
    )


def _write_auto(
    path: Path,
    *,
    run_id: str,
    rollouts: list[tuple[int, int, str]],
) -> None:
    """Write an auto_labels_<run_id>.json with the given rollouts."""
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "config_path": "configs/perturbation/spatial/act_spatial_y-5cm.yaml",
                "policy_id": "lerobot/act_aloha_sim_transfer_cube_human",
                "env_id": "gym_aloha/AlohaTransferCube-v0",
                "perturbation_kind": "spatial",
                "perturbation_params": {"dx_m": 0.0, "dy_m": -0.05},
                "perturbation_applied": True,
                "n_rollouts": len(rollouts),
                "distribution": {},
                "labels": [
                    {
                        "seed_group": sg,
                        "rollout_idx": ri,
                        "episode_seed": ri,
                        "failure_mode": label,
                        "evidence": {},
                    }
                    for sg, ri, label in rollouts
                ],
            }
        )
    )


def test_score_one_perfect_agreement_passes(tmp_path: Path) -> None:
    """All manual labels match auto → κ = 1.0 → PASS."""
    manual = [(0, i, "recovery_failure") for i in range(10)] + [
        (1, i, "success") for i in range(10)
    ]
    auto = manual  # identical
    _write_sample(
        tmp_path / "relabel_sample_X.json",
        run_id="X",
        unlock_at="2026-05-22T00:00:00+00:00",  # already past
        rollouts=manual,
    )
    _write_auto(tmp_path / "auto_labels_X.json", run_id="X", rollouts=auto)
    kappa = score_one(
        tmp_path / "relabel_sample_X.json",
        now=_dt.datetime(2026, 6, 1, tzinfo=_dt.UTC),
    )
    assert kappa == pytest.approx(1.0)


def test_score_one_blocks_before_unlock(tmp_path: Path) -> None:
    """Before unlock_at the scorer refuses (preserves blinding)."""
    rollouts = [(0, 0, "success")]
    _write_sample(
        tmp_path / "relabel_sample_Y.json",
        run_id="Y",
        unlock_at="2099-01-01T00:00:00+00:00",
        rollouts=rollouts,
    )
    _write_auto(tmp_path / "auto_labels_Y.json", run_id="Y", rollouts=rollouts)
    with pytest.raises(SystemExit, match="Embargo not yet"):
        score_one(
            tmp_path / "relabel_sample_Y.json",
            now=_dt.datetime(2026, 6, 1, tzinfo=_dt.UTC),
        )


def test_score_one_blocks_when_manual_labels_missing(tmp_path: Path) -> None:
    """Unfilled manual labels are rejected with a helpful message."""
    rollouts: list[tuple[int, int, str]] = [(0, 0, None)]  # type: ignore[list-item]
    _write_sample(
        tmp_path / "relabel_sample_Z.json",
        run_id="Z",
        unlock_at="2026-05-22T00:00:00+00:00",
        rollouts=rollouts,
    )
    _write_auto(
        tmp_path / "auto_labels_Z.json",
        run_id="Z",
        rollouts=[(0, 0, "success")],
    )
    with pytest.raises(SystemExit, match="manual_failure_mode"):
        score_one(
            tmp_path / "relabel_sample_Z.json",
            now=_dt.datetime(2026, 6, 1, tzinfo=_dt.UTC),
        )


def test_score_one_partial_agreement_returns_intermediate_kappa(
    tmp_path: Path,
) -> None:
    """Mixed agreement → κ strictly between 0 and 1."""
    manual = [(0, i, "recovery_failure") for i in range(5)] + [
        (0, i, "success") for i in range(5, 10)
    ]
    auto = [(0, i, "recovery_failure") for i in range(7)] + [
        (0, i, "success") for i in range(7, 10)
    ]
    _write_sample(
        tmp_path / "relabel_sample_M.json",
        run_id="M",
        unlock_at="2026-05-22T00:00:00+00:00",
        rollouts=manual,
    )
    _write_auto(tmp_path / "auto_labels_M.json", run_id="M", rollouts=auto)
    kappa = score_one(
        tmp_path / "relabel_sample_M.json",
        now=_dt.datetime(2026, 6, 1, tzinfo=_dt.UTC),
    )
    assert 0 < kappa < 1
