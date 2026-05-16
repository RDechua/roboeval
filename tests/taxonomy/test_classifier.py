"""Tests for the rule-based failure-mode classifier."""

from __future__ import annotations

from roboeval.evaluation.types import RolloutResult
from roboeval.taxonomy import FailureMode, classify_rollout


def _r(**overrides) -> RolloutResult:
    """Build a RolloutResult with sensible defaults for classifier tests."""
    base = {
        "seed_group": 0,
        "rollout_idx": 0,
        "episode_seed": 0,
        "success": False,
        "success_custom": False,
        "success_step": None,
        "n_steps": 400,
        "max_reward": 0,
        "terminated": False,
        "truncated": True,
        "wall_time_s": 8.0,
        "final_cube_z": 0.0,
        "final_cube_x": 0.0,
        "final_cube_y": 0.0,
        "final_cube_xy_dist": 0.0,
        "failure_mode": "",
    }
    base.update(overrides)
    return RolloutResult(**base)  # type: ignore[arg-type]


def test_primary_success_returns_none_label():
    label = classify_rollout(_r(success=True, terminated=True, n_steps=120))
    assert label.failure_mode is None
    assert label.evidence["success"] is True


def test_truncated_non_success_is_timeout():
    label = classify_rollout(
        _r(success=False, truncated=True, terminated=False, n_steps=400)
    )
    assert label.failure_mode == FailureMode.TIMEOUT
    assert label.evidence["timeout_rule"] == "truncated_without_terminated"


def test_terminated_non_success_routes_to_needs_review():
    # Episode terminated (gym-aloha terminated for some other reason — failure
    # category that isn't Timeout but isn't yet detectable until trajectory
    # data lands).
    label = classify_rollout(
        _r(success=False, terminated=True, truncated=False, n_steps=180)
    )
    assert label.failure_mode == FailureMode.NEEDS_REVIEW
    assert (
        "trajectory-data classifier not yet implemented"
        in label.evidence["needs_review_reason"]
    )


def test_success_custom_alone_is_still_primary_failure():
    # Custom-success but primary=False is an interesting case: the cube made
    # it into the calibrated zone but the gripper-grasp signal never fired.
    # Until trajectory data lands, this routes to NEEDS_REVIEW so the
    # human can decide between Grasp Failure (PRD §7.2) and a noise case.
    label = classify_rollout(
        _r(success=False, success_custom=True, terminated=True, n_steps=300)
    )
    assert label.failure_mode == FailureMode.NEEDS_REVIEW


def test_evidence_carries_signals_used():
    label = classify_rollout(
        _r(
            success=False,
            truncated=True,
            terminated=False,
            max_reward=2,
            final_cube_z=0.06,
        )
    )
    assert label.evidence["max_reward"] == 2
    assert label.evidence["final_cube_z"] == 0.06
    assert label.evidence["truncated"] is True


def test_failure_mode_enum_round_trips_as_string():
    # The enum inherits from str so it serialises naturally to JSON.
    assert FailureMode.TIMEOUT == "timeout"
    assert FailureMode.NEEDS_REVIEW.value == "needs_review"
    assert FailureMode.GRASP_FAILURE.value == "grasp_failure"


def test_rollout_label_carries_identifiers():
    label = classify_rollout(_r(seed_group=2, rollout_idx=17, episode_seed=200023))
    assert label.seed_group == 2
    assert label.rollout_idx == 17
    assert label.episode_seed == 200023
