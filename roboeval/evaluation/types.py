"""Typed result schemas for rollouts and evaluation runs.

The dataclasses here are the single source of truth for what a "rollout" and
an "evaluation" produce — every other module imports from this file rather
than defining its own structure. See PRD Section 6.3 (Metrics) for the
metric definitions; per-seed-group aggregation matches the PRD requirement
of "mean ± std across 3 random seeds and >=50 rollouts per condition".
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RolloutResult:
    """Outcome of a single rollout episode.

    Attributes:
        seed_group: Index of the PRD "seed group" this rollout belongs to
            (typically 0, 1, or 2 for a 3-seed eval).
        rollout_idx: Within-group rollout index, e.g. 0..49 for 50-per-seed.
        episode_seed: Actual seed passed to ``env.reset(seed=...)``;
            ``seed_group * 100_003 + rollout_idx`` by convention.
        success: Primary success signal — ``info["is_success"]`` from
            gym-aloha (``reward == 4``, matches model-card baseline).
        success_custom: Secondary success signal — PRD geometric criterion
            via :class:`roboeval.envs.success.TransferCubeSuccessDetector`.
        success_step: 1-based step number at which a success signal first
            fired (either primary or secondary), or ``None`` if neither did.
        n_steps: Total env steps executed before termination/truncation.
        max_reward: Maximum dm_control reward seen in this episode (0..4).
        terminated: Whether the env terminated (success or failure).
        truncated: Whether the env hit ``max_episode_steps``.
        wall_time_s: Wall-clock seconds for this rollout (post-reset).
        final_cube_z: Cube z-coordinate at end-of-episode (metres).
        final_cube_x: Cube x-coordinate at end-of-episode (metres).
            Used by the Week 2.5 ``calibrate`` subcommand to derive
            ``target_xy`` for the geometric success criterion.
        final_cube_y: Cube y-coordinate at end-of-episode (metres).
        final_cube_xy_dist: Euclidean distance from origin in xy at
            end-of-episode (metres); kept for backward-compatible
            diagnostics. ``sqrt(final_cube_x**2 + final_cube_y**2)``
            is the same number when ``target_xy=(0,0)``.
        failure_mode: Free-form label written by the failure-taxonomy
            classifier (Week 5). Empty string for Week 2 baseline runs.
        action_sign_flip_rate: Fraction of ``(step, action_dim)`` pairs at
            which the action sign flipped between consecutive steps —
            i.e. ``mean(sign(a_t) * sign(a_{t-1}) < 0)`` over all
            timesteps and dims. ``0.0`` for episodes shorter than two
            steps. Drives the PRD §7.2 Oscillation rule.
        terminal_eef_xy_distance_m: Minimum xy distance (metres) between
            the cube and either gripper-link body at the final step.
            ``None`` if dm_control physics isn't exposed (mock envs).
            Drives the PRD §7.2 Approach Failure rule.
        contact_made: Whether the cube ever touched a gripper finger at
            any point during the episode. Drives the PRD §7.2 Grasp
            Failure rule (contact without lift).
        last_50_step_cube_displacement_m: Euclidean xy displacement of the
            cube between the step ``min(50, n_steps)`` before the end and
            the final step (metres). Drives the PRD §7.2 Recovery /
            quiescence rule.
    """

    seed_group: int
    rollout_idx: int
    episode_seed: int
    success: bool
    success_custom: bool
    success_step: int | None
    n_steps: int
    max_reward: int
    terminated: bool
    truncated: bool
    wall_time_s: float
    final_cube_z: float
    final_cube_x: float
    final_cube_y: float
    final_cube_xy_dist: float
    failure_mode: str = ""
    action_sign_flip_rate: float = 0.0
    terminal_eef_xy_distance_m: float | None = None
    contact_made: bool = False
    last_50_step_cube_displacement_m: float = 0.0


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Aggregated metrics across all rollouts in one eval run.

    Aggregation policy: TSR is averaged *within* each seed group, then mean
    and population standard deviation are taken *across* seed groups. This
    matches the PRD 6.3 reporting convention ("mean ± std across 3 random
    seeds"), where the unit of variation is the seed group, not the
    individual rollout.

    Attributes:
        policy_id: HF repo id (e.g. ``lerobot/act_aloha_sim_transfer_cube_human``).
        env_id: Gymnasium env id (e.g. ``gym_aloha/AlohaTransferCube-v0``).
        rollouts: Immutable tuple of per-rollout results.
        mean_tsr: Mean across seed groups of primary TSR.
        std_tsr: Population stdev across seed groups of primary TSR.
        mean_tsr_custom: Same as ``mean_tsr`` for the geometric criterion.
        std_tsr_custom: Same as ``std_tsr`` for the geometric criterion.
        median_tts: Median ``success_step`` across rollouts that succeeded
            on the primary signal; ``None`` if zero successes.
        n_rollouts: Total number of rollouts across all seed groups.
        n_seed_groups: Number of distinct seed groups.
        per_seed_tsr: Per-seed-group primary TSRs (length ``n_seed_groups``).
        per_seed_tsr_custom: Per-seed-group secondary TSRs.
    """

    policy_id: str
    env_id: str
    rollouts: tuple[RolloutResult, ...]
    mean_tsr: float
    std_tsr: float
    mean_tsr_custom: float
    std_tsr_custom: float
    median_tts: float | None
    n_rollouts: int
    n_seed_groups: int
    per_seed_tsr: tuple[float, ...] = field(default_factory=tuple)
    per_seed_tsr_custom: tuple[float, ...] = field(default_factory=tuple)


def aggregate(
    rollouts: Sequence[RolloutResult],
    policy_id: str,
    env_id: str,
) -> EvalResult:
    """Aggregate per-rollout results into an :class:`EvalResult`.

    Computes TSR within each seed group first, then takes mean and population
    standard deviation across seed groups. Median TTS is taken over the set
    of rollouts that triggered the primary success signal.

    Args:
        rollouts: Per-rollout outcomes from a single eval run; must not be empty.
        policy_id: HuggingFace repo id of the policy.
        env_id: Gymnasium env id.

    Returns:
        Aggregated :class:`EvalResult`.

    Raises:
        ValueError: If ``rollouts`` is empty.
    """
    if not rollouts:
        raise ValueError("aggregate() requires at least one rollout")

    by_seed: dict[int, list[RolloutResult]] = {}
    for r in rollouts:
        by_seed.setdefault(r.seed_group, []).append(r)

    per_seed_tsr: list[float] = []
    per_seed_tsr_custom: list[float] = []
    for _seed, group in sorted(by_seed.items()):
        per_seed_tsr.append(sum(r.success for r in group) / len(group))
        per_seed_tsr_custom.append(sum(r.success_custom for r in group) / len(group))

    mean_tsr = statistics.fmean(per_seed_tsr)
    std_tsr = statistics.pstdev(per_seed_tsr) if len(per_seed_tsr) > 1 else 0.0
    mean_tsr_c = statistics.fmean(per_seed_tsr_custom)
    std_tsr_c = (
        statistics.pstdev(per_seed_tsr_custom) if len(per_seed_tsr_custom) > 1 else 0.0
    )

    tts_values = [
        r.success_step for r in rollouts if r.success and r.success_step is not None
    ]
    median_tts = float(statistics.median(tts_values)) if tts_values else None

    return EvalResult(
        policy_id=policy_id,
        env_id=env_id,
        rollouts=tuple(rollouts),
        mean_tsr=mean_tsr,
        std_tsr=std_tsr,
        mean_tsr_custom=mean_tsr_c,
        std_tsr_custom=std_tsr_c,
        median_tts=median_tts,
        n_rollouts=len(rollouts),
        n_seed_groups=len(by_seed),
        per_seed_tsr=tuple(per_seed_tsr),
        per_seed_tsr_custom=tuple(per_seed_tsr_custom),
    )
