"""Command-line entry point for RoboEval.

Week 1 ships a single subcommand, ``roboeval smoke``, which executes a
short random-action rollout against the ``gym_aloha/AlohaTransferCube-v0``
environment to validate the dependency stack. The done criterion for PRD
Section 10.2 Week 1 is "first rollout renders without crash"; this CLI is
the smallest artifact that satisfies it.

The full evaluation CLI (``roboeval evaluate config=...``) lands in
Week 2-3 when the evaluation engine is built (PRD Section 5.1).

Heavy dependencies (``gymnasium``, ``gym_aloha``, ``torch``) are imported
lazily inside :func:`_cmd_smoke` so that importing :mod:`roboeval.cli`
itself never requires them — this keeps unit tests and ``mypy --strict``
runnable in a minimal CI environment.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

_LOG = logging.getLogger("roboeval.cli")

_SMOKE_ENV_ID = "gym_aloha/AlohaTransferCube-v0"


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for the CLI.

    Args:
        verbose: When ``True``, emit ``DEBUG``-level logs; otherwise ``INFO``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cmd_smoke(rollout_steps: int) -> int:
    """Run a Week 1 random-action smoke rollout against gym-aloha.

    Uses ``env.action_space.sample()`` — policy inference is deliberately
    deferred to Week 2, when the policy loader subpackage lands. Success
    means the rollout completes (terminates or exhausts ``rollout_steps``)
    without raising.

    Args:
        rollout_steps: Maximum number of environment steps before exit.

    Returns:
        ``0`` on success, ``1`` if a required dependency is missing, or
        ``2`` if the rollout itself raises.
    """
    try:
        import gym_aloha  # noqa: F401  # registers the Aloha envs with gymnasium
        import gymnasium as gym
        import torch
    except ImportError as exc:
        _LOG.error("Missing dependency: %s", exc)
        _LOG.error("Run `uv pip install -e '.[dev]'` to install the stack.")
        return 1

    _LOG.info(
        "torch=%s mps_available=%s",
        torch.__version__,
        torch.backends.mps.is_available(),
    )

    try:
        env = gym.make(_SMOKE_ENV_ID)
        _LOG.info("created env %s", _SMOKE_ENV_ID)
        env.reset(seed=0)
        for step in range(rollout_steps):
            action = env.action_space.sample()
            _obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                _LOG.info(
                    "episode ended at step %d (terminated=%s truncated=%s)",
                    step,
                    bool(terminated),
                    bool(truncated),
                )
                break
        env.close()
    except Exception as exc:  # noqa: BLE001 - top-level smoke boundary
        _LOG.exception("rollout crashed: %s", exc)
        return 2

    _LOG.info("smoke rollout completed (max %d steps)", rollout_steps)
    return 0


def _cmd_evaluate(config_path: str) -> int:
    """Run a Hydra-configured ACT evaluation against gym-aloha.

    Loads the YAML config, instantiates the policy and env via
    :mod:`roboeval.policies.act_loader` and :mod:`roboeval.envs.aloha`,
    runs the multi-seed evaluation loop, logs to W&B (respecting the
    config's ``wandb.mode`` so smoke runs can be offline), and prints the
    final mean ± std TSR to stdout.

    Args:
        config_path: Path to the Hydra YAML config (e.g.
            ``configs/baseline/act_nominal_fast.yaml``).

    Returns:
        ``0`` on success, ``1`` if a dependency is missing, ``2`` if
        policy loading fails, ``3`` if the eval itself crashes.
    """
    from datetime import datetime

    try:
        from omegaconf import OmegaConf

        from roboeval.envs.aloha import ALOHA_TRANSFER_CUBE_ID, make_aloha_env
        from roboeval.envs.success import (
            SuccessCriterion,
            TransferCubeSuccessDetector,
        )
        from roboeval.evaluation.logger import wandb_run
        from roboeval.evaluation.loop import evaluate_policy
        from roboeval.policies.act_loader import load_act_policy
    except ImportError as exc:
        _LOG.error("Missing dependency: %s", exc)
        _LOG.error("Run `uv pip install -e '.[dev]'` to install the stack.")
        return 1

    cfg = OmegaConf.load(config_path)
    target_xy_list = list(cfg.success.target_xy)
    criterion = SuccessCriterion(
        z_threshold_m=float(cfg.success.z_threshold_m),
        xy_tolerance_m=float(cfg.success.xy_tolerance_m),
        dwell_steps=int(cfg.success.dwell_steps),
        target_xy=(float(target_xy_list[0]), float(target_xy_list[1])),
    )

    _LOG.info(
        "Loading ACT policy %s on device=%s",
        str(cfg.policy.repo_id),
        str(cfg.policy.device),
    )
    try:
        policy = load_act_policy(
            repo_id=str(cfg.policy.repo_id),
            task=str(cfg.env.task),
            device=str(cfg.policy.device),
        )
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Failed to load policy: %s", exc)
        return 2

    _LOG.info("Policy loaded on device=%s", policy.device)

    run_name = f"{cfg.wandb.name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config_dict = {
        "policy_id": str(cfg.policy.repo_id),
        "env_id": ALOHA_TRANSFER_CUBE_ID,
        "device": policy.device,
        "n_rollouts_per_seed": int(cfg.eval.n_rollouts_per_seed),
        "seeds": list(cfg.eval.seeds),
        "max_steps": int(cfg.eval.max_steps),
        "success_z_threshold_m": criterion.z_threshold_m,
        "success_xy_tolerance_m": criterion.xy_tolerance_m,
        "success_dwell_steps": criterion.dwell_steps,
        "success_target_xy": list(criterion.target_xy),
        "episode_length": int(cfg.env.episode_length),
        "lerobot_version": "0.4.4",
    }

    try:
        with wandb_run(
            project=str(cfg.wandb.project),
            name=run_name,
            config=config_dict,
            tags=list(cfg.wandb.tags),
            mode=str(cfg.wandb.mode),
        ) as handle:
            if handle.url:
                _LOG.info("W&B run URL: %s", handle.url)

            result = evaluate_policy(
                env_factory=lambda: make_aloha_env(
                    task=str(cfg.env.task),
                    episode_length=int(cfg.env.episode_length),
                ),
                policy=policy,
                detector_factory=lambda: TransferCubeSuccessDetector(criterion),
                seeds=list(cfg.eval.seeds),
                n_rollouts_per_seed=int(cfg.eval.n_rollouts_per_seed),
                max_steps=int(cfg.eval.max_steps),
                policy_id=str(cfg.policy.repo_id),
                env_id=ALOHA_TRANSFER_CUBE_ID,
                on_rollout=handle.log_rollout,
            )
            handle.log_summary(result)

            summary = (
                f"\n[roboeval] Evaluation complete.\n"
                f"  mean_tsr        = {result.mean_tsr:.3f} +/- {result.std_tsr:.3f}"
                f"  (primary, gym-aloha native is_success)\n"
                f"  mean_tsr_custom = "
                f"{result.mean_tsr_custom:.3f} +/- {result.std_tsr_custom:.3f}"
                f"  (PRD z+xy+dwell)\n"
                f"  median_tts      = {result.median_tts}\n"
                f"  n_rollouts      = {result.n_rollouts} across "
                f"{result.n_seed_groups} seed group(s)\n"
                f"  per_seed_tsr    = {result.per_seed_tsr}\n"
            )
            print(summary)
            if handle.url:
                print(f"  wandb_run_url   = {handle.url}")
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Evaluation crashed: %s", exc)
        return 3

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser` with the ``smoke``
        subcommand registered.
    """
    parser = argparse.ArgumentParser(
        prog="roboeval",
        description="RoboEval CLI (see docs/PRD.md).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Run a Week 1 dependency-stack smoke rollout.",
    )
    smoke.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Maximum random-action env steps to execute (default: 10).",
    )

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Run a Hydra-configured ACT evaluation (Week 2+).",
    )
    evaluate.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/baseline/act_nominal_fast.yaml).",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the RoboEval CLI.

    Args:
        argv: CLI arguments excluding the program name. Defaults to
            ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (``0`` on success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    _configure_logging(bool(args.verbose))

    if args.command == "smoke":
        return _cmd_smoke(int(args.steps))
    if args.command == "evaluate":
        return _cmd_evaluate(str(args.config))
    # Unreachable: subparsers(required=True) enforces a known command.
    raise AssertionError(f"unhandled command: {args.command!r}")
