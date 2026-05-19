"""PPO training loop wrapper for residual RL (PRD §8.2).

Thin glue between :mod:`roboeval.residual` primitives and
Stable-Baselines3's ``PPO``. The training environment is built by
composing:

1. The bare ALOHA env (``make_aloha_env``).
2. The chosen perturbation (typically ``spatial`` with ``dy_m=0.05``
   for the +5 cm Phase-4 base-policy target).
3. The frozen base policy (loaded via ``load_policy``).
4. :class:`ResidualEnvWrapper`, which composes base + residual and
   replaces the env's sparse reward with the configured shaping.

SB3's ``MlpPolicy`` provides the policy network. The architectural
choice of "2 hidden layers x 256 GELU" (PRD §8.2) is enforced via
``policy_kwargs``. The compositor's learnable ``alpha`` lives **outside**
SB3's optimised parameters in v1: PPO updates the MLP weights; alpha is
fixed at its initialisation per training run. Making alpha jointly
trainable requires a custom SB3 policy class (Week 7 follow-up).

Heavy dependencies (``stable_baselines3``, ``torch``) are imported
lazily inside :func:`train_residual` so the module is import-safe
without them.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym

from roboeval.policies.base import Policy
from roboeval.residual.env_wrapper import (
    FeatureExtractor,
    ResidualEnvWrapper,
    RewardFn,
    zero_feature_extractor,
)
from roboeval.residual.policy import ResidualCompositor

_LOG = logging.getLogger("roboeval.residual.train")

_DEFAULT_HIDDEN_DIM: int = 256
"""PRD §8.2 hidden dim for the SB3 MlpPolicy net_arch."""


def build_training_env(
    base_env_factory: Callable[[], gym.Env[Any, Any]],
    base_policy: Policy,
    compositor: ResidualCompositor,
    reward_fn: RewardFn,
    feature_extractor: FeatureExtractor = zero_feature_extractor,
) -> ResidualEnvWrapper:
    """Build the residual-action training env.

    Pulled out as its own function so unit tests can construct the
    same wrapper without going through SB3.

    Args:
        base_env_factory: Zero-arg callable returning the underlying
            ALOHA env (typically already wrapped with a perturbation).
        base_policy: Frozen base policy.
        compositor: Residual compositor instance.
        reward_fn: Reward function.
        feature_extractor: Obs → feature-vector callable.

    Returns:
        The wrapped env, ready for SB3 PPO.
    """
    env = base_env_factory()
    return ResidualEnvWrapper(
        env=env,
        base_policy=base_policy,
        compositor=compositor,
        reward_fn=reward_fn,
        feature_extractor=feature_extractor,
    )


def train_residual(
    base_env_factory: Callable[[], gym.Env[Any, Any]],
    base_policy: Policy,
    compositor: ResidualCompositor,
    reward_fn: RewardFn,
    output_dir: Path | str,
    *,
    total_timesteps: int = 500_000,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    seed: int = 0,
    verbose: int = 1,
    log_std_init: float = 0.0,
    feature_extractor: FeatureExtractor = zero_feature_extractor,
) -> Path:
    """Run SB3 PPO against a residual-wrapped env; save the trained model.

    PRD §8.2 default training budget is 500k timesteps.

    Args:
        base_env_factory: Zero-arg callable building the underlying env.
        base_policy: Frozen base policy (e.g. ACT).
        compositor: Compositor instance (its alpha is fixed at init for v1).
        reward_fn: Per-step reward function.
        output_dir: Directory to save the trained PPO model into.
        total_timesteps: Env-step training budget.
        learning_rate: SB3 PPO learning rate.
        n_steps: SB3 PPO rollout length per update.
        batch_size: SB3 PPO minibatch size.
        n_epochs: SB3 PPO update epochs per rollout.
        gamma: Discount factor.
        seed: SB3 PPO RNG seed.
        verbose: SB3 verbose level.
        log_std_init: Initial log-std of PPO's Gaussian policy. SB3's
            default ``0.0`` (std=1.0) combined with our compositor's
            ``alpha=0.1`` perturbs the base by up to ±0.2 per dim,
            which destroys ACT's narrow successful trajectory on the
            +5cm cell. ``-2.0`` (std≈0.14) keeps the initial residual
            small enough to preserve the base TSR while PPO bootstraps.
        feature_extractor: Obs feature extractor for the residual MLP
            input. Default is the zero-width extractor; production use
            substitutes an ACT-encoder hook (Week 7).

    Returns:
        Path to the saved PPO model directory.
    """
    from stable_baselines3 import PPO

    env = build_training_env(
        base_env_factory=base_env_factory,
        base_policy=base_policy,
        compositor=compositor,
        reward_fn=reward_fn,
        feature_extractor=feature_extractor,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG.info(
        "Starting PPO residual training: total_timesteps=%d, lr=%.3g, "
        "n_steps=%d, log_std_init=%.2f, output_dir=%s",
        total_timesteps,
        learning_rate,
        n_steps,
        log_std_init,
        str(out_dir),
    )

    policy_kwargs: dict[str, Any] = {
        "net_arch": [_DEFAULT_HIDDEN_DIM, _DEFAULT_HIDDEN_DIM],
        "log_std_init": log_std_init,
    }
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        seed=seed,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
    )
    model.learn(total_timesteps=total_timesteps)
    save_path = out_dir / "ppo_residual"
    model.save(str(save_path))
    _LOG.info("Saved trained PPO residual to %s", str(save_path))
    return save_path


__all__ = ["build_training_env", "train_residual"]
