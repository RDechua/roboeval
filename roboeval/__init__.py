"""RoboEval: evaluation harness and failure-mode study for robot learning policies.

This is the top-level package for the RoboEval project. The authoritative
specification lives in ``docs/PRD.md``; the package layout follows PRD
Section 5.2 (Repository Structure).

Subpackages:
    envs: Environment wrappers conforming to the Gymnasium API.
    policies: Policy loaders and inference wrappers.
    evaluation: Rollout engine and metric collectors.
    taxonomy: Failure-mode classifier.
    residual: Residual RL trainer (Stable-Baselines3 PPO).
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
