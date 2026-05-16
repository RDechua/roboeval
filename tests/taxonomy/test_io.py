"""Tests for the auto-labels JSON writer (PRD §7.3 step 4)."""

from __future__ import annotations

import json

import pytest

from roboeval.taxonomy import (
    SCHEMA_VERSION,
    FailureMode,
    RolloutLabel,
    compute_distribution,
    labels_to_json_obj,
    write_auto_labels,
)


def _label(idx: int, mode: FailureMode | None) -> RolloutLabel:
    return RolloutLabel(
        seed_group=0,
        rollout_idx=idx,
        episode_seed=idx,
        failure_mode=mode,
        evidence={"rule": "test"},
    )


def test_compute_distribution_counts_each_bucket():
    labels = [
        _label(0, None),
        _label(1, None),
        _label(2, FailureMode.TIMEOUT),
        _label(3, FailureMode.GRASP_FAILURE),
        _label(4, FailureMode.GRASP_FAILURE),
    ]
    dist = compute_distribution(labels)
    assert dist["success"] == 2
    assert dist["timeout"] == 1
    assert dist["grasp_failure"] == 2
    # Unseen buckets are present with count 0 (downstream-dashboard ergonomics).
    assert dist["approach_failure"] == 0
    assert dist["action_oscillation"] == 0
    assert dist["recovery_failure"] == 0
    assert dist["needs_review"] == 0
    assert dist["visual_confusion"] == 0


def test_compute_distribution_sums_to_n_rollouts():
    labels = [_label(i, FailureMode.TIMEOUT) for i in range(7)]
    dist = compute_distribution(labels)
    assert sum(dist.values()) == 7


def test_labels_to_json_obj_matches_schema():
    labels = [_label(0, None), _label(1, FailureMode.TIMEOUT)]
    obj = labels_to_json_obj(
        labels,
        run_id="abc123",
        config_path="configs/baseline/act_nominal.yaml",
        policy_id="lerobot/act",
        env_id="gym_aloha/AlohaTransferCube-v0",
        perturbation_kind="spatial",
        perturbation_params={"dy_m": 0.01},
        perturbation_applied=True,
    )
    assert obj["schema_version"] == SCHEMA_VERSION
    assert obj["run_id"] == "abc123"
    assert obj["perturbation_kind"] == "spatial"
    assert obj["perturbation_params"] == {"dy_m": 0.01}
    assert obj["perturbation_applied"] is True
    assert obj["n_rollouts"] == 2
    assert obj["distribution"]["success"] == 1
    assert obj["distribution"]["timeout"] == 1
    assert obj["labels"][0]["failure_mode"] is None
    assert obj["labels"][1]["failure_mode"] == "timeout"


def test_write_auto_labels_creates_file_and_dir(tmp_path):
    out_dir = tmp_path / "data" / "taxonomy"
    labels = [_label(0, None), _label(1, FailureMode.GRASP_FAILURE)]
    path = write_auto_labels(
        labels,
        output_dir=out_dir,
        run_id="run42",
        config_path="configs/x.yaml",
        policy_id="lerobot/act",
        env_id="gym_aloha/AlohaTransferCube-v0",
        perturbation_kind="none",
        perturbation_params={},
        perturbation_applied=False,
    )
    assert path == out_dir / "auto_labels_run42.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["run_id"] == "run42"
    assert data["n_rollouts"] == 2
    assert len(data["labels"]) == 2


def test_write_auto_labels_round_trip_preserves_evidence(tmp_path):
    labels = [
        RolloutLabel(
            seed_group=1,
            rollout_idx=4,
            episode_seed=100007,
            failure_mode=FailureMode.RECOVERY_FAILURE,
            evidence={
                "rule": "perturbed_quiet_policy_and_stalled_cube",
                "action_sign_flip_rate": 0.012,
                "perturbation_applied": True,
            },
        )
    ]
    path = write_auto_labels(
        labels,
        output_dir=tmp_path,
        run_id="r",
        config_path="c",
        policy_id="p",
        env_id="e",
        perturbation_kind="spatial",
        perturbation_params={"dy_m": 0.03},
        perturbation_applied=True,
    )
    obj = json.loads(path.read_text())
    only = obj["labels"][0]
    assert only["seed_group"] == 1
    assert only["rollout_idx"] == 4
    assert only["episode_seed"] == 100007
    assert only["failure_mode"] == "recovery_failure"
    assert only["evidence"]["rule"] == "perturbed_quiet_policy_and_stalled_cube"
    assert only["evidence"]["action_sign_flip_rate"] == pytest.approx(0.012)
    assert only["evidence"]["perturbation_applied"] is True


def test_write_auto_labels_empty_labels_writes_zero_distribution(tmp_path):
    path = write_auto_labels(
        [],
        output_dir=tmp_path,
        run_id="empty",
        config_path="c",
        policy_id="p",
        env_id="e",
        perturbation_kind="none",
        perturbation_params={},
        perturbation_applied=False,
    )
    obj = json.loads(path.read_text())
    assert obj["n_rollouts"] == 0
    assert obj["labels"] == []
    assert sum(obj["distribution"].values()) == 0
