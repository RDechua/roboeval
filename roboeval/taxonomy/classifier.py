"""Rule-based failure-mode classifier (PRD §7.2 / §7.3).

:func:`classify_rollout` takes one :class:`RolloutResult` (and an optional
run-level ``perturbation_applied`` flag) and returns ``None`` for a
primary-successful rollout or a :class:`FailureMode` for a primary-failed
one.

Rule priority (first match wins)
--------------------------------
The PRD §7.3 says rollouts matching zero or multiple categories go to
``NEEDS_REVIEW``. In practice the operational rules overlap (e.g. a
contact-made grasp failure is also truncated and stalled at the end), so
this v1.1 classifier resolves ties by priority rather than declaring a
review case for every overlap. The priority order encodes which proximate
cause we report:

1. **ACTION_OSCILLATION** — motor-level pathology, fires regardless of
   outcome. Most specific failure mode.
2. **GRASP_FAILURE** — contact made but cube never lifted.
3. **APPROACH_FAILURE** — no contact AND EE far from cube xy at end.
4. **RECOVERY_FAILURE** — only when caller passes
   ``perturbation_applied=True``: quiet policy that stalled out the cube.
   The flag is run-level (the perturbation kind comes from the config),
   not per-rollout, so it isn't on :class:`RolloutResult`.
5. **TIMEOUT** — truncated AND cube displacement < 1 cm in the last
   50 steps (PRD "no progress" half of the Timeout rule).
6. **NEEDS_REVIEW** — fall-through bucket for rollouts whose signals
   didn't trigger any v1.1 rule.

``VISUAL_CONFUSION`` is an aggregate-over-runs rule and is emitted by a
separate analyser when comparing nominal vs visually-perturbed run pairs.
It is not produced by this per-rollout classifier.
"""

from __future__ import annotations

from roboeval.evaluation.types import RolloutResult
from roboeval.taxonomy.types import FailureMode, RolloutLabel

_OSCILLATION_FLIP_RATE_MIN: float = 0.40
"""Episode-mean sign-flip rate above which we call ACTION_OSCILLATION.

PRD §7.2 specifies a windowed rule (">5 flips per 10-step window for
>5 steps"); we approximate it with the episode-level mean because
:class:`RolloutResult` carries only the aggregate. ``0.40`` sits just
below the ``0.50`` expected for sign-uniform random actions and well
above typical smooth-policy values (< 0.10 in nominal ACT rollouts).
Empirical tuning may move this once we have labelled data.
"""

_GRASP_CUBE_Z_MAX_M: float = 0.02
"""PRD §7.2 Grasp Failure: terminal cube z below this counts as 'not lifted'."""

_APPROACH_EEF_DIST_MIN_M: float = 0.05
"""PRD §7.2 Approach Failure: EE-cube xy distance above this is 'wrong position'."""

_RECOVERY_FLIP_RATE_MAX: float = 0.05
"""Sign-flip rate below which the policy is considered 'quiet'.

Proxy for PRD §7.2 Recovery Failure's 'action variance post-perturbation
< 0.1' — the variance signal isn't yet on :class:`RolloutResult`. A quiet
policy in our data has flip rate < 0.05; the threshold may tighten as we
gather post-perturbation-only statistics.
"""

_TIMEOUT_DISPLACEMENT_MAX_M: float = 0.01
"""PRD §7.2 Timeout: last-50-step cube xy displacement below this is 'no progress'."""


def classify_rollout(
    rollout: RolloutResult,
    *,
    perturbation_applied: bool = False,
) -> RolloutLabel:
    """Classify one rollout into a failure mode (or success → None).

    Args:
        rollout: One per-rollout result from the eval loop.
        perturbation_applied: Run-level flag — ``True`` when the eval
            config applied any perturbation (spatial / visual / dynamic /
            temporal). Required for the Recovery rule to fire; ignored
            by all other rules.

    Returns:
        A :class:`RolloutLabel` with ``failure_mode=None`` if the
        rollout's primary success flag is ``True``; otherwise the
        detected :class:`FailureMode` (or ``NEEDS_REVIEW`` if no v1.1
        rule triggered).
    """
    evidence: dict[str, object] = {
        "success": rollout.success,
        "success_custom": rollout.success_custom,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "n_steps": rollout.n_steps,
        "max_reward": rollout.max_reward,
        "final_cube_z": rollout.final_cube_z,
        "action_sign_flip_rate": rollout.action_sign_flip_rate,
        "terminal_eef_xy_distance_m": rollout.terminal_eef_xy_distance_m,
        "contact_made": rollout.contact_made,
        "last_50_step_cube_displacement_m": rollout.last_50_step_cube_displacement_m,
        "perturbation_applied": perturbation_applied,
    }

    def _label(mode: FailureMode | None, **extra: object) -> RolloutLabel:
        evidence.update(extra)
        return RolloutLabel(
            seed_group=rollout.seed_group,
            rollout_idx=rollout.rollout_idx,
            episode_seed=rollout.episode_seed,
            failure_mode=mode,
            evidence=evidence,
        )

    if rollout.success:
        return _label(None)

    # 1. ACTION_OSCILLATION — motor-level pathology, fires regardless of
    # outcome so a thrashing policy is labelled before its downstream
    # consequence (no contact, dropped cube, timeout, etc.).
    if rollout.action_sign_flip_rate > _OSCILLATION_FLIP_RATE_MIN:
        return _label(
            FailureMode.ACTION_OSCILLATION,
            rule="action_sign_flip_rate_above_threshold",
        )

    # 2. GRASP_FAILURE — contact made at some point AND cube never lifted.
    # Stricter than "primary failure" alone: catches the "touched but
    # dropped" mode that motivates the Phase 4 residual RL focus.
    if rollout.contact_made and rollout.final_cube_z < _GRASP_CUBE_Z_MAX_M:
        return _label(
            FailureMode.GRASP_FAILURE,
            rule="contact_made_and_cube_not_lifted",
        )

    # 3. APPROACH_FAILURE — never made contact AND terminal EE is far
    # from cube xy. Requires the gripper-pose accessor (mock envs return
    # None and this rule simply doesn't fire — falling through to
    # NEEDS_REVIEW, which is the correct behaviour for missing signal).
    if (
        not rollout.contact_made
        and rollout.terminal_eef_xy_distance_m is not None
        and rollout.terminal_eef_xy_distance_m > _APPROACH_EEF_DIST_MIN_M
    ):
        return _label(
            FailureMode.APPROACH_FAILURE,
            rule="no_contact_and_eef_far_from_cube",
        )

    # 4. RECOVERY_FAILURE — perturbation applied AND quiet policy AND
    # stalled cube. The "quiet policy" proxy (low sign-flip rate) stands
    # in for PRD §7.2's "action variance < 0.1" until we surface variance
    # on the rollout schema.
    if (
        perturbation_applied
        and rollout.action_sign_flip_rate < _RECOVERY_FLIP_RATE_MAX
        and rollout.last_50_step_cube_displacement_m < _TIMEOUT_DISPLACEMENT_MAX_M
    ):
        return _label(
            FailureMode.RECOVERY_FAILURE,
            rule="perturbed_quiet_policy_and_stalled_cube",
        )

    # 5. TIMEOUT — truncated AND cube made no progress in last 50 steps.
    # The "no progress" half uses the new trajectory aggregate; before
    # Week 5 every truncated non-success matched this, which was
    # uninformative.
    if (
        rollout.truncated
        and rollout.last_50_step_cube_displacement_m < _TIMEOUT_DISPLACEMENT_MAX_M
    ):
        return _label(
            FailureMode.TIMEOUT,
            rule="truncated_and_no_progress_last_50_steps",
        )

    return _label(
        FailureMode.NEEDS_REVIEW,
        needs_review_reason=(
            "no rule matched: trajectory signals didn't trigger any v1.1 "
            "detection rule (PRD §7.3 manual review bucket)"
        ),
    )


__all__ = ["classify_rollout"]
