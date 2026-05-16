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


def test_truncated_non_success_with_stalled_cube_is_timeout():
    label = classify_rollout(
        _r(
            success=False,
            truncated=True,
            terminated=False,
            n_steps=400,
            last_50_step_cube_displacement_m=0.0,
        )
    )
    assert label.failure_mode == FailureMode.TIMEOUT
    assert label.evidence["rule"] == "truncated_and_no_progress_last_50_steps"


def test_terminated_non_success_routes_to_needs_review():
    # terminated=True with no contact, no EE signal, no oscillation falls
    # through every v1.1 rule and lands in NEEDS_REVIEW.
    label = classify_rollout(
        _r(success=False, terminated=True, truncated=False, n_steps=180)
    )
    assert label.failure_mode == FailureMode.NEEDS_REVIEW
    assert "no rule matched" in label.evidence["needs_review_reason"]


def test_success_custom_alone_routes_through_trajectory_rules():
    # Custom-success but primary=False with no contact and no EE info
    # falls through to NEEDS_REVIEW. With richer trajectory signals
    # (contact_made, EE distance) the same rollout would be classified
    # as Grasp or Approach. truncated=False here so the TIMEOUT rule
    # doesn't pick this up first.
    label = classify_rollout(
        _r(
            success=False,
            success_custom=True,
            terminated=True,
            truncated=False,
            n_steps=300,
        )
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


# --- Week 5: trajectory-driven rules ---------------------------------------


def test_high_sign_flip_rate_is_action_oscillation():
    label = classify_rollout(_r(action_sign_flip_rate=0.6, truncated=True))
    assert label.failure_mode == FailureMode.ACTION_OSCILLATION
    assert label.evidence["rule"] == "action_sign_flip_rate_above_threshold"


def test_contact_made_with_unlifted_cube_is_grasp_failure():
    label = classify_rollout(
        _r(
            contact_made=True,
            final_cube_z=0.005,
            truncated=True,
            last_50_step_cube_displacement_m=0.0,
        )
    )
    assert label.failure_mode == FailureMode.GRASP_FAILURE
    assert label.evidence["rule"] == "contact_made_and_cube_not_lifted"


def test_no_contact_far_eef_is_approach_failure():
    label = classify_rollout(
        _r(
            contact_made=False,
            terminal_eef_xy_distance_m=0.10,
            truncated=True,
            last_50_step_cube_displacement_m=0.0,
        )
    )
    assert label.failure_mode == FailureMode.APPROACH_FAILURE
    assert label.evidence["rule"] == "no_contact_and_eef_far_from_cube"


def test_approach_rule_does_not_fire_when_eef_distance_unknown():
    # terminal_eef_xy_distance_m=None (mock env). Falls through to TIMEOUT
    # because truncated=True and the cube didn't move.
    label = classify_rollout(
        _r(
            contact_made=False,
            terminal_eef_xy_distance_m=None,
            truncated=True,
            last_50_step_cube_displacement_m=0.0,
        )
    )
    assert label.failure_mode == FailureMode.TIMEOUT


def test_perturbed_quiet_stalled_policy_is_recovery_failure():
    label = classify_rollout(
        _r(
            action_sign_flip_rate=0.0,
            last_50_step_cube_displacement_m=0.0,
            truncated=True,
        ),
        perturbation_applied=True,
    )
    assert label.failure_mode == FailureMode.RECOVERY_FAILURE
    assert label.evidence["rule"] == "perturbed_quiet_policy_and_stalled_cube"


def test_recovery_rule_requires_perturbation_applied_flag():
    # Same trajectory signals as the recovery test, but without the flag —
    # should fall through to TIMEOUT.
    label = classify_rollout(
        _r(
            action_sign_flip_rate=0.0,
            last_50_step_cube_displacement_m=0.0,
            truncated=True,
        ),
        perturbation_applied=False,
    )
    assert label.failure_mode == FailureMode.TIMEOUT


def test_oscillation_takes_priority_over_grasp():
    # A thrashing policy that also happens to contact the cube and leave
    # it on the table should label as OSCILLATION (the motor pathology
    # is the more specific finding).
    label = classify_rollout(
        _r(
            action_sign_flip_rate=0.6,
            contact_made=True,
            final_cube_z=0.005,
            truncated=True,
        )
    )
    assert label.failure_mode == FailureMode.ACTION_OSCILLATION


def test_grasp_takes_priority_over_timeout():
    # Grasp signature dominates a TIMEOUT-eligible truncated/stalled rollout.
    label = classify_rollout(
        _r(
            contact_made=True,
            final_cube_z=0.005,
            truncated=True,
            last_50_step_cube_displacement_m=0.0,
        )
    )
    assert label.failure_mode == FailureMode.GRASP_FAILURE


def test_recovery_takes_priority_over_timeout_when_perturbed():
    # Quiet policy + stalled cube + perturbation flag → RECOVERY beats TIMEOUT.
    label = classify_rollout(
        _r(
            action_sign_flip_rate=0.0,
            last_50_step_cube_displacement_m=0.0,
            truncated=True,
        ),
        perturbation_applied=True,
    )
    assert label.failure_mode == FailureMode.RECOVERY_FAILURE


def test_truncated_with_moving_cube_does_not_match_timeout():
    # Cube was moving at the end → not "no progress" → TIMEOUT shouldn't fire.
    # With no contact, no oscillation, no EE signal, falls to NEEDS_REVIEW.
    label = classify_rollout(
        _r(
            truncated=True,
            last_50_step_cube_displacement_m=0.05,  # well above threshold
        )
    )
    assert label.failure_mode == FailureMode.NEEDS_REVIEW
