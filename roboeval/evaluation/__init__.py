"""Rollout engine and metric collectors.

Implements the reproducible eval harness described in PRD Section 5.1: given a
Hydra config, runs N rollouts across M seeds and produces the metrics listed
in PRD Section 6.3 (TSR, TTS, perturbation recovery rate, failure-mode
distribution, residual-RL delta, eval reproducibility sigma).
"""

from __future__ import annotations
