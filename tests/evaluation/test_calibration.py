"""Unit tests for the target_xy calibration math and config resolver."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from roboeval.evaluation.calibration import (
    _MIN_SUCCESSES_REQUIRED,
    _PERCENTILE,
    calibrate_target_xy,
    calibration_to_dict,
    clear_calibration_cache,
    load_calibration_json,
    register_calibration_resolver,
)
from roboeval.evaluation.types import RolloutResult


def _r(idx, *, success, x=0.0, y=0.0):
    return RolloutResult(
        seed_group=0,
        rollout_idx=idx,
        episode_seed=idx,
        success=success,
        success_custom=False,
        success_step=10 if success else None,
        n_steps=10,
        max_reward=4 if success else 0,
        terminated=success,
        truncated=not success,
        wall_time_s=0.5,
        final_cube_z=0.1 if success else 0.0,
        final_cube_x=x,
        final_cube_y=y,
        final_cube_xy_dist=math.hypot(x, y),
    )


def test_calibrate_returns_centroid_and_90th_percentile():
    # 30 successful rollouts clustered around (0.5, 0.0) with mm-scale noise.
    # x range: 0.485..0.514 (3cm spread), y range: 0..0.004 (4mm spread)
    rollouts = [
        _r(i, success=True, x=0.5 + 0.001 * (i - 15), y=0.0 + 0.001 * (i % 5))
        for i in range(30)
    ]
    result = calibrate_target_xy(rollouts)
    assert result.target_xy[0] == pytest.approx(0.5, abs=1e-3)
    assert result.target_xy[1] == pytest.approx(0.002, abs=1e-3)
    assert result.n_rollouts == 30
    assert result.n_successes == 30
    assert result.percentile == _PERCENTILE
    # 90th percentile of a ~3cm-spread cluster is ~1-2cm
    assert 0.005 < result.xy_tolerance_m < 0.02


def test_calibrate_excludes_failures_from_centroid():
    rollouts = [_r(i, success=True, x=0.5, y=0.0) for i in range(25)]
    # Failures with wildly different xy must NOT pull the centroid
    rollouts += [_r(100 + i, success=False, x=100.0, y=-100.0) for i in range(5)]
    result = calibrate_target_xy(rollouts)
    assert result.target_xy == pytest.approx((0.5, 0.0))
    assert result.n_successes == 25
    assert result.n_rollouts == 30


def test_calibrate_refuses_below_minimum_successes():
    n = _MIN_SUCCESSES_REQUIRED - 1
    rollouts = [_r(i, success=True, x=0.5, y=0.0) for i in range(n)]
    with pytest.raises(ValueError, match="primary successes"):
        calibrate_target_xy(rollouts)


def test_calibrate_refuses_empty():
    with pytest.raises(ValueError, match="at least one rollout"):
        calibrate_target_xy([])


def test_calibration_to_dict_round_trips_provenance():
    rollouts = [_r(i, success=True, x=0.5, y=0.0) for i in range(30)]
    result = calibrate_target_xy(rollouts)
    payload = calibration_to_dict(
        result,
        git_sha="abc1234",
        timestamp="2026-05-12T00:00:00Z",
        source_config="configs/baseline/act_nominal.yaml",
        policy_id="lerobot/act_aloha_sim_transfer_cube_human",
        env_id="gym_aloha/AlohaTransferCube-v0",
    )
    assert payload["target_xy"] == [pytest.approx(0.5), pytest.approx(0.0)]
    assert payload["git_sha"] == "abc1234"
    assert payload["n_successes"] == 30
    assert payload["percentile"] == _PERCENTILE
    assert len(payload["success_endpoints"]) == 30
    # Endpoints must be plain-JSON-serializable lists, not tuples
    assert isinstance(payload["success_endpoints"][0], list)


# ---------------------------------------------------------------------------
# load_calibration_json + register_calibration_resolver
# ---------------------------------------------------------------------------


def _write_calibration(path: Path, *, target_xy, xy_tolerance_m, extras=None):
    payload = {
        "target_xy": list(target_xy),
        "xy_tolerance_m": xy_tolerance_m,
    }
    if extras:
        payload.update(extras)
    path.write_text(json.dumps(payload))


def test_load_calibration_json_returns_parsed_dict(tmp_path):
    calib = tmp_path / "calib.json"
    _write_calibration(
        calib,
        target_xy=(-0.018, 0.506),
        xy_tolerance_m=0.022,
        extras={"git_sha": "deadbee", "n_successes": 44},
    )
    data = load_calibration_json(calib)
    assert data["target_xy"] == [-0.018, 0.506]
    assert data["xy_tolerance_m"] == 0.022
    assert data["git_sha"] == "deadbee"


def test_load_calibration_json_missing_file_helpful_error(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(FileNotFoundError, match="roboeval calibrate"):
        load_calibration_json(missing)


def test_load_calibration_json_rejects_missing_required_keys(tmp_path):
    calib = tmp_path / "broken.json"
    calib.write_text(json.dumps({"target_xy": [0, 0]}))  # missing xy_tolerance_m
    with pytest.raises(KeyError, match="xy_tolerance_m"):
        load_calibration_json(calib)


def test_calibration_resolver_interpolates_in_omegaconf(tmp_path):
    from omegaconf import OmegaConf

    calib = tmp_path / "calib.json"
    _write_calibration(calib, target_xy=(-0.018, 0.506), xy_tolerance_m=0.022)
    clear_calibration_cache()
    register_calibration_resolver(path=calib)

    cfg = OmegaConf.create(
        {
            "success": {
                "target_xy": "${calibration:target_xy}",
                "xy_tolerance_m": "${calibration:xy_tolerance_m}",
            }
        }
    )
    assert list(cfg.success.target_xy) == [-0.018, 0.506]
    assert cfg.success.xy_tolerance_m == 0.022


def test_calibration_resolver_unknown_key_raises(tmp_path):
    from omegaconf import OmegaConf

    calib = tmp_path / "calib.json"
    _write_calibration(calib, target_xy=(0, 0), xy_tolerance_m=0.01)
    clear_calibration_cache()
    register_calibration_resolver(path=calib)

    cfg = OmegaConf.create({"x": "${calibration:not_a_real_key}"})
    with pytest.raises(Exception, match="not_a_real_key"):
        _ = cfg.x
