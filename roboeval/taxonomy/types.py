"""Failure-mode taxonomy data model (PRD §7.2).

The six failure categories below are operationally defined by the
detection rules in PRD §7.2 so that any two labellers agree on the
category for a given rollout. ``NEEDS_REVIEW`` is the edge-case bucket
for rollouts that match zero or multiple categories (PRD §7.3).

``classify_rollout`` (in :mod:`roboeval.taxonomy.classifier`) returns
``None`` for primary-successful rollouts and one of these enum values
for primary-failed rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureMode(str, Enum):
    """Operational failure-mode labels from PRD §7.2.

    The enum inherits from ``str`` so values serialise as plain strings
    in the auto-/manual-label JSONs (PRD §7.3) without a custom encoder.
    """

    GRASP_FAILURE = "grasp_failure"
    """Robot contacts object but drops or misses. PRD detection rule:
    finger-object contact at any step AND terminal cube z < 0.02 m."""

    APPROACH_FAILURE = "approach_failure"
    """Robot reaches wrong position/orientation before contact. PRD
    detection rule: end-effector terminal pose error > 5 cm from cube
    COM AND no contact in episode."""

    RECOVERY_FAILURE = "recovery_failure"
    """Policy cannot correct after perturbation. PRD detection rule:
    perturbation applied AND terminal TSR=0 AND action variance
    post-perturbation < 0.1."""

    ACTION_OSCILLATION = "action_oscillation"
    """Policy outputs rapidly alternating contradictory actions. PRD
    detection rule: action sign-flip rate > 5 per 10-step window for
    >5 steps."""

    TIMEOUT = "timeout"
    """Task not completed within step budget. PRD detection rule:
    episode hits step cap AND no progress (cube displacement < 1 cm)
    in last 50 steps. (The "no progress" half requires trajectory data
    not yet on RolloutResult; the implemented v1 rule treats any
    truncated, non-successful episode as a Timeout candidate.)"""

    VISUAL_CONFUSION = "visual_confusion"
    """Policy error correlates with visual change. PRD detection rule:
    TSR drops > 30% under changed visual condition AND TSR ≥ 70% in
    nominal. (Aggregate-over-runs rule; emitted by a separate
    after-the-fact analyser, not per-rollout classify_rollout.)"""

    NEEDS_REVIEW = "needs_review"
    """Rollout matched zero categories or multiple. PRD §7.3: flagged
    for manual review; the bucket size is reported alongside the
    failure-distribution heatmap."""


@dataclass(frozen=True, slots=True)
class RolloutLabel:
    """One classifier output for one rollout, plus the evidence used.

    Stored together with the auto-label JSON (PRD §7.3) so future
    re-labelling has access to the exact signals the rule-based
    classifier saw at decision time.

    Attributes:
        seed_group: Source rollout's seed group (matches
            :class:`roboeval.evaluation.types.RolloutResult.seed_group`).
        rollout_idx: Within-group rollout index.
        episode_seed: ``env.reset`` seed (for de-duplication when joining
            labels back to the rollouts table).
        failure_mode: The category, or ``None`` for a primary-successful
            rollout that doesn't need a failure label.
        evidence: Free-form dict of the signals the classifier consulted
            (e.g. ``{"terminated": True, "truncated": False, "max_reward": 1}``)
            so a human re-labeller can audit the auto-decision.
    """

    seed_group: int
    rollout_idx: int
    episode_seed: int
    failure_mode: FailureMode | None
    evidence: dict[str, Any] = field(default_factory=dict)


__all__ = ["FailureMode", "RolloutLabel"]
