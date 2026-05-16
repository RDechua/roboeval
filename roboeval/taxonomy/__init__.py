"""Failure-mode classifier.

Operationalises the six-category failure taxonomy from PRD Section 7.2
(Grasp Failure, Approach Failure, Recovery Failure, Action Oscillation,
Timeout, Visual Confusion) into a rule-based classifier with manual review
hooks.

Public surface:

* :class:`roboeval.taxonomy.types.FailureMode` — the seven-value enum
  (six categories + NEEDS_REVIEW).
* :class:`roboeval.taxonomy.types.RolloutLabel` — one classifier output.
* :func:`roboeval.taxonomy.classifier.classify_rollout` — rule-based
  classifier.
* :func:`roboeval.taxonomy.agreement.cohens_kappa` — inter-rater κ for
  the PRD §7.3 blinded self-relabel protocol (target κ > 0.6).
"""

from __future__ import annotations

from roboeval.taxonomy.agreement import KappaResult, cohens_kappa
from roboeval.taxonomy.classifier import classify_rollout
from roboeval.taxonomy.types import FailureMode, RolloutLabel

__all__ = [
    "FailureMode",
    "KappaResult",
    "RolloutLabel",
    "classify_rollout",
    "cohens_kappa",
]
