"""Residual RL for Phase 4 (PRD §8).

Public surface:

* :class:`ResidualMLP` — the trainable residual.
* :class:`ResidualCompositor` — combines base + alpha * residual.
* :class:`ResidualEnvWrapper` — gym wrapper that swaps the env's
  reward and composes the base policy with the residual.
* :func:`sparse_success_reward`, :func:`shaped_distance_reward`,
  :func:`combined_reward` — reward components.
* :func:`train_residual` — SB3 PPO training entry point.
"""

from __future__ import annotations

from roboeval.residual.env_wrapper import (
    FeatureExtractor,
    ResidualEnvWrapper,
    RewardFn,
    zero_feature_extractor,
)
from roboeval.residual.policy import ResidualCompositor, ResidualMLP
from roboeval.residual.reward import (
    combined_reward,
    shaped_distance_reward,
    sparse_success_reward,
)
from roboeval.residual.train import build_training_env, train_residual

__all__ = [
    "FeatureExtractor",
    "ResidualCompositor",
    "ResidualEnvWrapper",
    "ResidualMLP",
    "RewardFn",
    "build_training_env",
    "combined_reward",
    "shaped_distance_reward",
    "sparse_success_reward",
    "train_residual",
    "zero_feature_extractor",
]
