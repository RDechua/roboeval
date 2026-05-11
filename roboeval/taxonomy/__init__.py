"""Failure-mode classifier.

Operationalises the six-category failure taxonomy from PRD Section 7.2
(Grasp Failure, Approach Failure, Recovery Failure, Action Oscillation,
Timeout, Visual Confusion) into a rule-based classifier with manual review
hooks.
"""

from __future__ import annotations
