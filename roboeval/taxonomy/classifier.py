"""Rule-based failure-mode classifier (PRD §7.2 / §7.3).

:func:`classify_rollout` takes one :class:`RolloutResult` and returns
``None`` for a primary-successful rollout or a :class:`FailureMode`
for a primary-failed one.

What this v1 implements
-----------------------
The four categories whose detection rules need only data already on
``RolloutResult`` (n_steps, terminated, truncated, success, max_reward,
final_cube_*) are implemented:

* **TIMEOUT** — episode hit the step cap without succeeding.
* **NEEDS_REVIEW** — anything that doesn't match a precise rule yet.

What this v1 defers
-------------------
The remaining four categories (GRASP_FAILURE, APPROACH_FAILURE,
ACTION_OSCILLATION, RECOVERY_FAILURE) each need data that
``RolloutResult`` doesn't yet carry — finger-object contact bits,
end-effector terminal pose, per-step action vectors, post-perturbation
action variance. Adding those is an evaluation-loop change tracked in
the Week-5 plan; until then those branches fall through to
``NEEDS_REVIEW`` with the missing-data reason recorded in ``evidence``.

VISUAL_CONFUSION is an aggregate-over-runs rule, not per-rollout; it's
emitted by a separate analyser when comparing nominal vs visually-
perturbed run pairs and is not handled here.
"""

from __future__ import annotations

from roboeval.evaluation.types import RolloutResult
from roboeval.taxonomy.types import FailureMode, RolloutLabel

# PRD §7.2 timeout detection threshold for cube "no progress" check —
# kept here as a named constant so the Week-5 trajectory-data
# upgrade can reuse it without changing the rule's meaning.
_CUBE_NO_PROGRESS_DISPLACEMENT_M: float = 0.01
"""Cube displacement < 1 cm in the last 50 steps counts as 'no progress'."""


def classify_rollout(rollout: RolloutResult) -> RolloutLabel:
    """Classify one rollout into a failure mode (or success → None).

    Args:
        rollout: One per-rollout result from the eval loop.

    Returns:
        A :class:`RolloutLabel` with ``failure_mode=None`` if the
        rollout's primary success flag is ``True``; otherwise the
        detected :class:`FailureMode` (or ``NEEDS_REVIEW`` if the
        per-rollout signals don't match any implemented rule).
    """
    evidence: dict[str, object] = {
        "success": rollout.success,
        "success_custom": rollout.success_custom,
        "terminated": rollout.terminated,
        "truncated": rollout.truncated,
        "n_steps": rollout.n_steps,
        "max_reward": rollout.max_reward,
        "final_cube_z": rollout.final_cube_z,
    }

    # Successful rollouts don't get a failure label. The classifier's
    # job is to attribute failures, not duplicate the success column.
    if rollout.success:
        return RolloutLabel(
            seed_group=rollout.seed_group,
            rollout_idx=rollout.rollout_idx,
            episode_seed=rollout.episode_seed,
            failure_mode=None,
            evidence=evidence,
        )

    # TIMEOUT — primary failure AND truncated (episode hit step cap).
    # The PRD's "no progress in last 50 steps" half of the rule requires
    # per-step cube trajectory data; until that lands, treat any
    # truncated non-success as a Timeout candidate.
    if rollout.truncated and not rollout.terminated:
        evidence["timeout_rule"] = "truncated_without_terminated"
        return RolloutLabel(
            seed_group=rollout.seed_group,
            rollout_idx=rollout.rollout_idx,
            episode_seed=rollout.episode_seed,
            failure_mode=FailureMode.TIMEOUT,
            evidence=evidence,
        )

    # Everything else — the four trajectory-data-requiring categories —
    # falls through. The Week-5 trajectory-data extension fills these
    # in; until then, route them to NEEDS_REVIEW with the missing-data
    # reason recorded.
    evidence["needs_review_reason"] = (
        "non-timeout failure: trajectory-data classifier not yet implemented "
        "(grasp/approach/oscillation/recovery rules require per-step actions "
        "and contact bits not yet on RolloutResult)"
    )
    return RolloutLabel(
        seed_group=rollout.seed_group,
        rollout_idx=rollout.rollout_idx,
        episode_seed=rollout.episode_seed,
        failure_mode=FailureMode.NEEDS_REVIEW,
        evidence=evidence,
    )


__all__ = ["classify_rollout"]
