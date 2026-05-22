"""Tests for scripts.build_headline_json — the headline.json producer.

These tests synthesise minimal auto_labels + phase4_ablation fixtures in
``tmp_path`` so they run in CI (where the real artifacts under
``data/taxonomy/`` and ``outputs/`` are gitignored and absent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_headline_json import (
    _CONFIG_TO_CELL,
    _PHASE3_STATS,
    build_headline_payload,
)

# Inverse mapping: cell_id -> canonical config_path basename. Uses the
# first config_path that maps to each cell (script accepts both
# `_1step.yaml` and `_1steps.yaml` for the temporal cells).
_PREFERRED_CONFIG_FOR: dict[str, str] = {}
for cfg_name, cell_id in _CONFIG_TO_CELL.items():
    _PREFERRED_CONFIG_FOR.setdefault(cell_id, cfg_name)


# Three Phase 4 ablation run_ids — must match _ABLATION_RUNS in the script.
_ABLATION_RUN_IDS: dict[str, str] = {
    "A": "w6k2wole",
    "B": "o6ukyo53",
    "C": "43czuigy",
}


def _write_auto_labels(
    dir_: Path,
    *,
    run_id: str,
    config_path: str,
    distribution: dict[str, int],
) -> Path:
    """Write a minimal auto_labels JSON payload to ``dir_``."""
    dir_.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "config_path": config_path,
        "policy_id": "lerobot/act_aloha_sim_transfer_cube_human",
        "env_id": "gym_aloha/AlohaTransferCube-v0",
        "perturbation_kind": "spatial",
        "perturbation_params": {},
        "perturbation_applied": False,
        "n_rollouts": sum(distribution.values()),
        "distribution": distribution,
        "labels": [],
    }
    path = dir_ / f"auto_labels_{run_id}.json"
    path.write_text(json.dumps(payload))
    return path


def _default_distribution(success: int, recovery: int) -> dict[str, int]:
    """Pad a 2-bucket count to the full 8-key distribution shape."""
    return {
        "success": success,
        "grasp_failure": 0,
        "approach_failure": 0,
        "recovery_failure": recovery,
        "action_oscillation": 0,
        "timeout": 0,
        "visual_confusion": 0,
        "needs_review": 150 - success - recovery,
    }


def _seed_fixture_tree(repo_root: Path) -> None:
    """Create the minimum on-disk artifacts the script needs."""
    labels_dir = repo_root / "data" / "taxonomy"
    for cell_id in _PHASE3_STATS:
        _write_auto_labels(
            labels_dir,
            run_id=f"rid-{cell_id}",
            config_path=f"configs/perturbation/spatial/{_PREFERRED_CONFIG_FOR[cell_id]}",
            distribution=_default_distribution(success=30, recovery=90),
        )

    # Phase 4 ablation auto_labels (distinct run_ids matching _ABLATION_RUNS).
    _write_auto_labels(
        labels_dir,
        run_id=_ABLATION_RUN_IDS["A"],
        config_path="configs/perturbation/spatial/act_spatial_y+5cm.yaml",
        distribution=_default_distribution(48, 89),
    )
    _write_auto_labels(
        labels_dir,
        run_id=_ABLATION_RUN_IDS["B"],
        config_path="configs/residual/residual_ppo_y+5cm_sparse.yaml",
        distribution=_default_distribution(28, 106),
    )
    _write_auto_labels(
        labels_dir,
        run_id=_ABLATION_RUN_IDS["C"],
        config_path="configs/residual/residual_ppo_y+5cm_shaped.yaml",
        distribution=_default_distribution(32, 109),
    )

    # phase4_ablation.json (tracked in the real repo; synthesised here).
    ablation_dir = repo_root / "docs" / "figures"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    (ablation_dir / "phase4_ablation.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "target_perturbation": {"kind": "spatial", "dx_m": 0.0, "dy_m": 0.05},
                "conditions": [
                    {
                        "condition_id": "A",
                        "label": "Frozen base only",
                        "n_runs": 1,
                        "n_rollouts": 150,
                        "per_seed_means": [0.26, 0.4, 0.3],
                        "mean": 0.32,
                        "std": 0.0589,
                        "bootstrap_ci_low": 0.26,
                        "bootstrap_ci_high": 0.4,
                        "run_ids": [_ABLATION_RUN_IDS["A"]],
                    },
                    {
                        "condition_id": "B",
                        "label": "Residual RL, sparse reward",
                        "n_runs": 1,
                        "n_rollouts": 150,
                        "per_seed_means": [0.22, 0.18, 0.16],
                        "mean": 0.1867,
                        "std": 0.0249,
                        "bootstrap_ci_low": 0.16,
                        "bootstrap_ci_high": 0.22,
                        "run_ids": [_ABLATION_RUN_IDS["B"]],
                    },
                    {
                        "condition_id": "C",
                        "label": "Residual RL, shaped reward",
                        "n_runs": 1,
                        "n_rollouts": 150,
                        "per_seed_means": [0.28, 0.2, 0.16],
                        "mean": 0.2133,
                        "std": 0.0499,
                        "bootstrap_ci_low": 0.16,
                        "bootstrap_ci_high": 0.28,
                        "run_ids": [_ABLATION_RUN_IDS["C"]],
                    },
                ],
                "comparisons": [
                    {
                        "condition_id": "B",
                        "delta_tsr": -0.1333,
                        "t_statistic": -2.95,
                        "df": 2.7,
                        "p_value": 0.966,
                        "significant_at_05": False,
                    },
                    {
                        "condition_id": "C",
                        "delta_tsr": -0.1067,
                        "t_statistic": -1.95,
                        "df": 3.9,
                        "p_value": 0.938,
                        "significant_at_05": False,
                    },
                ],
                "warnings": [],
            }
        )
    )


def test_build_headline_payload_has_ten_cells(tmp_path: Path) -> None:
    """6 spatial + 3 temporal + nominal = 10 cells."""
    _seed_fixture_tree(tmp_path)
    payload = build_headline_payload(repo_root=tmp_path)
    assert payload["schema_version"] == 2
    cells = payload["cells"]
    assert len(cells) == 10
    axes = {c["axis"] for c in cells}
    assert axes == {"spatial", "temporal", "nominal"}


def test_build_headline_payload_failure_counts_sum_to_n_rollouts(
    tmp_path: Path,
) -> None:
    _seed_fixture_tree(tmp_path)
    payload = build_headline_payload(repo_root=tmp_path)
    for cell in payload["cells"]:
        counts = cell["failure_counts"]
        assert sum(counts.values()) == cell["n_rollouts"]


def test_build_headline_payload_spatial_cells_have_known_means(
    tmp_path: Path,
) -> None:
    """Mean/std come from the script's STATE.md transcription, not the
    auto_labels — verify the transcribed values survive the build."""
    _seed_fixture_tree(tmp_path)
    payload = build_headline_payload(repo_root=tmp_path)
    by_id = {c["cell_id"]: c for c in payload["cells"]}
    assert by_id["y-5cm"]["mean_tsr_custom"] == 0.127
    assert by_id["y+5cm"]["mean_tsr_custom"] == 0.307
    assert by_id["delay-5step"]["mean_tsr_custom"] == 0.687
    assert by_id["nominal"]["mean_tsr_custom"] == 0.800


def test_build_headline_payload_includes_ablation_block(tmp_path: Path) -> None:
    _seed_fixture_tree(tmp_path)
    payload = build_headline_payload(repo_root=tmp_path)
    assert "ablation" in payload
    assert {c["condition_id"] for c in payload["ablation"]} == {"A", "B", "C"}
    by_id = {c["condition_id"]: c for c in payload["ablation"]}
    assert by_id["B"]["failure_counts"]["recovery_failure"] == 106
    assert by_id["B"]["run_id"] == _ABLATION_RUN_IDS["B"]


def test_build_headline_payload_includes_welch_tests(tmp_path: Path) -> None:
    _seed_fixture_tree(tmp_path)
    payload = build_headline_payload(repo_root=tmp_path)
    assert "welch_tests" in payload
    arms = {w["arm_id"] for w in payload["welch_tests"]}
    assert arms == {"B", "C"}


def test_build_headline_payload_missing_ablation_raises(tmp_path: Path) -> None:
    """Without phase4_ablation.json, the build script should fail loudly."""
    # Seed only the per-cell auto_labels; intentionally skip phase4_ablation.json.
    labels_dir = tmp_path / "data" / "taxonomy"
    for cell_id in _PHASE3_STATS:
        _write_auto_labels(
            labels_dir,
            run_id=f"rid-{cell_id}",
            config_path=f"configs/perturbation/spatial/{_PREFERRED_CONFIG_FOR[cell_id]}",
            distribution=_default_distribution(success=30, recovery=90),
        )
    with pytest.raises(FileNotFoundError, match="phase4_ablation"):
        build_headline_payload(repo_root=tmp_path)


def test_headline_json_file_committed_and_valid() -> None:
    """The repository must contain a tracked data/headline.json
    matching schema v2 (cells + ablation + welch_tests)."""
    repo_root = Path(__file__).resolve().parents[2]
    headline = json.loads((repo_root / "data" / "headline.json").read_text())
    assert headline["schema_version"] == 2
    assert len(headline["cells"]) == 10
    assert len(headline["ablation"]) == 3
    assert len(headline["welch_tests"]) == 2
