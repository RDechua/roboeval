"""Tests for the PRD §7.3 step 4 relabel-sample exporter."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from export_relabel_sample import (  # noqa: E402
    DEFAULT_LOCK_DAYS,
    SCHEMA_VERSION,
    build_sample_obj,
    export_relabel_sample,
    stratified_sample,
)


def _label(idx: int, mode: str | None, seed_group: int = 0):
    return {
        "seed_group": seed_group,
        "rollout_idx": idx,
        "episode_seed": idx,
        "failure_mode": mode,
        "evidence": {"rule": "test"},
    }


def _auto_labels_obj(labels):
    return {
        "schema_version": 1,
        "run_id": "test_run",
        "config_path": "configs/x.yaml",
        "policy_id": "lerobot/act",
        "env_id": "gym_aloha/AlohaTransferCube-v0",
        "perturbation_kind": "spatial",
        "perturbation_params": {"dy_m": 0.05},
        "perturbation_applied": True,
        "n_rollouts": len(labels),
        "distribution": {},
        "labels": labels,
    }


def test_stratified_sample_takes_n_per_bucket():
    labels = (
        [_label(i, None) for i in range(20)]
        + [_label(20 + i, "recovery_failure") for i in range(15)]
        + [_label(35 + i, "approach_failure") for i in range(8)]
    )
    chosen = stratified_sample(labels, per_category_n=5, seed=42)
    # 3 buckets x 5 == 15.
    assert len(chosen) == 15
    by_mode = {None: 0, "recovery_failure": 0, "approach_failure": 0}
    for c in chosen:
        by_mode[c.get("failure_mode")] += 1
    assert by_mode == {None: 5, "recovery_failure": 5, "approach_failure": 5}


def test_stratified_sample_takes_all_when_bucket_smaller_than_n(capsys):
    labels = [_label(0, "approach_failure"), _label(1, "approach_failure")]
    chosen = stratified_sample(labels, per_category_n=5, seed=0)
    assert len(chosen) == 2
    err = capsys.readouterr().err
    assert "warning" in err.lower()
    assert "approach_failure" in err


def test_stratified_sample_is_deterministic_given_seed():
    labels = [_label(i, "recovery_failure") for i in range(50)]
    a = stratified_sample(labels, per_category_n=5, seed=123)
    b = stratified_sample(labels, per_category_n=5, seed=123)
    assert a == b


def test_stratified_sample_different_seeds_produce_different_choices():
    labels = [_label(i, "recovery_failure") for i in range(50)]
    a = stratified_sample(labels, per_category_n=5, seed=1)
    b = stratified_sample(labels, per_category_n=5, seed=2)
    # Very unlikely (P < 10^-7) that two 5-of-50 picks coincide exactly.
    assert a != b


def test_build_sample_obj_redacts_failure_mode(tmp_path):
    labels = [_label(i, "recovery_failure") for i in range(20)] + [
        _label(20 + i, None) for i in range(20)
    ]
    auto_obj = _auto_labels_obj(labels)
    auto_path = tmp_path / "auto_labels_test_run.json"
    auto_path.write_text(json.dumps(auto_obj))

    sample = build_sample_obj(
        auto_obj,
        auto_path,
        per_category_n=5,
        lock_days=7,
        now=datetime(2026, 5, 17, 12, 0, tzinfo=UTC),
    )
    assert sample["schema_version"] == SCHEMA_VERSION
    assert sample["per_category_n"] == 5
    assert sample["run_id"] == "test_run"
    assert len(sample["samples"]) == 10  # 2 buckets x 5
    for s in sample["samples"]:
        assert s["manual_failure_mode"] is None
        # The original auto-classifier label MUST NOT leak through.
        assert "failure_mode" not in s
        assert "evidence" not in s


def test_build_sample_obj_records_sha256_audit_trail(tmp_path):
    labels = [_label(i, None) for i in range(10)]
    auto_obj = _auto_labels_obj(labels)
    auto_path = tmp_path / "auto_labels_test_run.json"
    auto_path.write_text(json.dumps(auto_obj))

    sample = build_sample_obj(
        auto_obj,
        auto_path,
        per_category_n=5,
        lock_days=7,
        now=datetime(2026, 5, 17, 12, 0, tzinfo=UTC),
    )
    sha = sample["auto_labels_sha256"]
    assert isinstance(sha, str)
    assert len(sha) == 64
    # Different content → different hash (regression check).
    other = _auto_labels_obj([_label(99, None)])
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other))
    other_sample = build_sample_obj(
        other,
        other_path,
        per_category_n=5,
        lock_days=7,
        now=datetime(2026, 5, 17, 12, 0, tzinfo=UTC),
    )
    assert other_sample["auto_labels_sha256"] != sha


def test_build_sample_obj_unlock_is_n_days_after_now():
    labels = [_label(i, None) for i in range(10)]
    auto_obj = _auto_labels_obj(labels)
    now = datetime(2026, 5, 17, 12, 0, tzinfo=UTC)
    sample = build_sample_obj(
        auto_obj,
        Path("/dev/null"),  # SHA path; will hash an empty/missing file path
        per_category_n=5,
        lock_days=10,
        now=now,
    )
    exported = datetime.fromisoformat(sample["exported_at"])
    unlock = datetime.fromisoformat(sample["unlock_at"])
    assert unlock - exported == timedelta(days=10)


def test_export_relabel_sample_writes_both_files(tmp_path):
    labels = [_label(i, None) for i in range(10)] + [
        _label(10 + i, "recovery_failure") for i in range(10)
    ]
    auto_obj = _auto_labels_obj(labels)
    input_path = tmp_path / "auto_labels_test_run.json"
    input_path.write_text(json.dumps(auto_obj))

    sample_path, unlock_path = export_relabel_sample(
        input_path,
        tmp_path / "out",
        per_category_n=3,
        lock_days=7,
        now=datetime(2026, 5, 17, tzinfo=UTC),
    )
    assert sample_path.exists()
    assert unlock_path.exists()
    sample = json.loads(sample_path.read_text())
    assert sample["per_category_n"] == 3
    assert len(sample["samples"]) == 6  # 2 buckets x 3
    # Unlock sidecar file holds just the ISO timestamp.
    assert unlock_path.read_text().strip() == sample["unlock_at"]


def test_export_relabel_sample_default_lock_is_at_least_seven_days():
    """PRD §7.3 step 4 minimum is 7 days; the default must satisfy it."""
    assert DEFAULT_LOCK_DAYS >= 7


def test_sample_identifiers_match_source_labels(tmp_path):
    labels = [_label(i, "recovery_failure", seed_group=i // 5) for i in range(20)]
    auto_obj = _auto_labels_obj(labels)
    input_path = tmp_path / "auto_labels_test_run.json"
    input_path.write_text(json.dumps(auto_obj))
    sample_path, _ = export_relabel_sample(
        input_path,
        tmp_path / "out",
        per_category_n=5,
        lock_days=7,
        now=datetime(2026, 5, 17, tzinfo=UTC),
    )
    sample = json.loads(sample_path.read_text())
    # Every sampled identifier must correspond to a real label in the input.
    sampled_keys = {
        (s["seed_group"], s["rollout_idx"], s["episode_seed"])
        for s in sample["samples"]
    }
    source_keys = {
        (label["seed_group"], label["rollout_idx"], label["episode_seed"])
        for label in labels
    }
    assert sampled_keys.issubset(source_keys)
