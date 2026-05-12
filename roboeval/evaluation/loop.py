"""Multi-rollout, multi-seed evaluation driver.

Implements the PRD Section 6.3 reporting convention: for each base seed
group, run ``n_rollouts_per_seed`` rollouts with deterministic episode
seeds ``seed_group * 100_003 + rollout_idx``; aggregate TSR within each
group, then report mean ± std across groups.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym

from roboeval.envs.aloha import get_cube_state
from roboeval.envs.success import TransferCubeSuccessDetector
from roboeval.evaluation.rollout import CubeStateFn, run_rollout
from roboeval.evaluation.types import EvalResult, RolloutResult, aggregate
from roboeval.policies.base import Policy

_LOG = logging.getLogger("roboeval.evaluation.loop")

_SEED_GROUP_OFFSET = 100_003
"""Large prime gap between seed groups so per-rollout seeds don't collide."""


def evaluate_policy(
    env_factory: Callable[[], gym.Env[Any, Any]],
    policy: Policy,
    detector_factory: Callable[[], TransferCubeSuccessDetector],
    seeds: Sequence[int],
    n_rollouts_per_seed: int,
    max_steps: int,
    policy_id: str,
    env_id: str,
    on_rollout: Callable[[RolloutResult], None] | None = None,
    cube_state_fn: CubeStateFn | None = None,
) -> EvalResult:
    """Iterate ``(seed_group, rollout_idx)`` pairs and aggregate the results.

    A single env is reused across all rollouts (each ``env.reset(seed=...)``
    re-randomises the initial conditions). The success detector is
    constructed once via ``detector_factory`` and reset internally between
    rollouts.

    Args:
        env_factory: Zero-arg callable returning a freshly built env. Called
            exactly once per evaluation run.
        policy: Policy to evaluate.
        detector_factory: Zero-arg callable returning a configured
            :class:`TransferCubeSuccessDetector`. Called exactly once;
            reset between rollouts.
        seeds: Sequence of seed-group identifiers (typically ``[0, 1, 2]``
            per PRD Section 6.3).
        n_rollouts_per_seed: Rollouts per seed group (PRD: ≥50).
        max_steps: Step cap per rollout (PRD Section 7.2: 400).
        policy_id: HuggingFace repo id, for the EvalResult.
        env_id: Gymnasium env id, for the EvalResult.
        on_rollout: Optional per-rollout callback (used by the W&B logger
            to stream rows into the rollouts table as the eval progresses).
        cube_state_fn: Accessor for the cube's 7-element qpos slice. When
            ``None`` (the default) we look up
            :func:`roboeval.envs.aloha.get_cube_state` at call time — this
            indirection is what lets unit tests monkeypatch
            ``roboeval.evaluation.loop.get_cube_state`` and have the patch
            take effect (binding through a default argument captures the
            original function object at definition time and is unaffected
            by later ``setattr``).

    Returns:
        Aggregated :class:`EvalResult`.
    """
    if cube_state_fn is None:
        cube_state_fn = get_cube_state
    env = env_factory()
    detector = detector_factory()
    results: list[RolloutResult] = []

    n_total = len(seeds) * n_rollouts_per_seed
    _LOG.info(
        "Starting evaluation: %d seed group(s) x %d rollouts = %d total",
        len(seeds),
        n_rollouts_per_seed,
        n_total,
    )

    try:
        for seed_group in seeds:
            for idx in range(n_rollouts_per_seed):
                episode_seed = seed_group * _SEED_GROUP_OFFSET + idx
                result = run_rollout(
                    env=env,
                    policy=policy,
                    success_detector=detector,
                    seed_group=seed_group,
                    rollout_idx=idx,
                    episode_seed=episode_seed,
                    max_steps=max_steps,
                    cube_state_fn=cube_state_fn,
                )
                results.append(result)
                if on_rollout is not None:
                    on_rollout(result)
                _LOG.info(
                    "seed_group=%d rollout=%d success=%s success_custom=%s "
                    "steps=%d wall=%.1fs",
                    seed_group,
                    idx,
                    result.success,
                    result.success_custom,
                    result.n_steps,
                    result.wall_time_s,
                )
    finally:
        env.close()

    return aggregate(results, policy_id=policy_id, env_id=env_id)
