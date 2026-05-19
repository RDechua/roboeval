"""Residual RL for Phase 4 (PRD §8).

Public surface:

* :class:`ResidualMLP` — the trainable residual.
* :class:`ResidualCompositor` — combines base + alpha * residual.
* :class:`ResidualEnvWrapper` — gym wrapper that swaps the env's
  reward and composes the base policy with the residual.
* :func:`sparse_success_reward`, :func:`shaped_distance_reward`,
  :func:`combined_reward` — reward components.
* :func:`train_residual` — SB3 PPO training entry point.
* :func:`aggregate_runs`, :class:`AblationReport` — PRD §8.3 ablation
  aggregator over persisted ``eval_results_<run_id>.json`` artifacts.
"""

from __future__ import annotations

from roboeval.residual.aggregate import (
    AblationReport,
    ConditionComparison,
    ConditionStats,
    aggregate_runs,
    classify_condition,
    format_markdown,
    load_eval_results,
    report_to_dict,
)
from roboeval.residual.composite import ResidualCompositePolicy
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
    "AblationReport",
    "ConditionComparison",
    "ConditionStats",
    "FeatureExtractor",
    "ResidualCompositePolicy",
    "ResidualCompositor",
    "ResidualEnvWrapper",
    "ResidualMLP",
    "RewardFn",
    "aggregate_runs",
    "build_training_env",
    "classify_condition",
    "combined_reward",
    "format_markdown",
    "load_eval_results",
    "report_to_dict",
    "shaped_distance_reward",
    "sparse_success_reward",
    "train_residual",
    "zero_feature_extractor",
]
