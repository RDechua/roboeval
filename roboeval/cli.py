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
from typing import Any

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
        from roboeval.envs.aloha import ALOHA_TRANSFER_CUBE_ID, make_aloha_env
        from roboeval.envs.perturb import make_perturbed_env
        from roboeval.envs.success import (
            SuccessCriterion,
            TransferCubeSuccessDetector,
        )
        from roboeval.evaluation.calibration import register_calibration_resolver
        from roboeval.evaluation.config import load_eval_config
        from roboeval.evaluation.logger import wandb_run
        from roboeval.evaluation.loop import evaluate_policy
        from roboeval.policies.factory import load_policy
        from roboeval.taxonomy import (
            classify_rollout,
            compute_distribution,
            write_auto_labels,
        )
    except ImportError as exc:
        _LOG.error("Missing dependency: %s", exc)
        _LOG.error("Run `uv pip install -e '.[dev]'` to install the stack.")
        return 1

    # Register the `${calibration:...}` resolver so configs can interpolate
    # frozen calibration values directly. Done before load_eval_config so
    # interpolation is resolved on access.
    register_calibration_resolver()

    cfg = load_eval_config(config_path)
    target_xy_list = list(cfg.success.target_xy)
    criterion = SuccessCriterion(
        z_threshold_m=float(cfg.success.z_threshold_m),
        xy_tolerance_m=float(cfg.success.xy_tolerance_m),
        dwell_steps=int(cfg.success.dwell_steps),
        target_xy=(float(target_xy_list[0]), float(target_xy_list[1])),
    )

    _LOG.info(
        "Loading %s policy %s on device=%s",
        str(cfg.policy.kind),
        str(cfg.policy.repo_id),
        str(cfg.policy.device),
    )
    try:
        policy = load_policy(
            kind=str(cfg.policy.kind),
            repo_id=str(cfg.policy.repo_id),
            task=str(cfg.env.task),
            device=str(cfg.policy.device),
        )
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Failed to load policy: %s", exc)
        return 2

    _LOG.info("Policy loaded on device=%s", policy.device)

    # Optional perturbation block. When absent (or kind=='none') the env
    # factory is the bare ALOHA env; when present we compose a wrapper
    # via roboeval.envs.perturb.make_perturbed_env. Recording the kind +
    # full param dict in the W&B config keeps every (axis, intensity)
    # cell self-describing on the dashboard.
    perturb_cfg = cfg.get("perturbation")
    perturb_kind: str | None = None
    perturb_params: dict[str, Any] = {}
    if perturb_cfg is not None and str(perturb_cfg.get("kind", "none")) != "none":
        perturb_kind = str(perturb_cfg.kind)
        perturb_params = {k: v for k, v in dict(perturb_cfg).items() if k != "kind"}

    def _env_factory() -> Any:
        env = make_aloha_env(
            task=str(cfg.env.task),
            episode_length=int(cfg.env.episode_length),
        )
        if perturb_kind is not None:
            return make_perturbed_env(env, kind=perturb_kind, **perturb_params)
        return env

    run_name = f"{cfg.wandb.name_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    config_dict = {
        "policy_kind": str(cfg.policy.kind),
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
        "perturbation_kind": perturb_kind or "none",
        "perturbation_params": perturb_params,
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
                env_factory=_env_factory,
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

            # PRD §7.3 steps 2 + 4: auto-classify every rollout, write
            # the frozen labels artifact, and surface the distribution.
            perturbation_applied = perturb_kind is not None
            labels = [
                classify_rollout(r, perturbation_applied=perturbation_applied)
                for r in result.rollouts
            ]
            distribution = compute_distribution(labels)
            run_id = str(handle.run_id) if handle.run_id is not None else run_name
            labels_path = write_auto_labels(
                labels,
                output_dir="data/taxonomy",
                run_id=run_id,
                config_path=str(config_path),
                policy_id=str(cfg.policy.repo_id),
                env_id=ALOHA_TRANSFER_CUBE_ID,
                perturbation_kind=perturb_kind or "none",
                perturbation_params=perturb_params,
                perturbation_applied=perturbation_applied,
            )
            handle.log_distribution(distribution)

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
                f"  failure_dist    = {distribution}\n"
                f"  auto_labels     = {labels_path}\n"
            )
            print(summary)
            if handle.url:
                print(f"  wandb_run_url   = {handle.url}")
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Evaluation crashed: %s", exc)
        return 3

    return 0


def _cmd_calibrate(
    config_path: str,
    output_path: str,
    n_rollouts: int,
) -> int:
    """Calibrate ``target_xy`` and ``xy_tolerance_m`` from N nominal rollouts.

    Runs the ACT policy in nominal conditions, collects the cube xy
    endpoint for every primary-successful rollout, and writes a frozen
    calibration JSON. The operator then updates the success block in
    the eval configs by hand (or via a subsequent commit) using the
    values printed at the end of the run.

    See :mod:`roboeval.evaluation.calibration` for the math.

    Args:
        config_path: Hydra YAML used as the source-of-truth for
            ``policy``, ``env``, ``eval.max_steps``, and ``wandb``
            blocks. The ``eval.seeds`` and ``eval.n_rollouts_per_seed``
            fields are overridden by ``--n-rollouts`` so the caller
            doesn't need a separate config file.
        output_path: Where to write the calibration JSON (typically
            ``data/calibration/transfer_cube_target_xy.json``).
        n_rollouts: Number of single-seed rollouts to run.

    Returns:
        ``0`` on success, ``1`` if a dependency is missing, ``2`` if
        the policy fails to load, ``3`` if calibration fails to derive
        a centroid (e.g. too few successes), ``4`` if the rollouts
        themselves crash.
    """
    import json
    from datetime import UTC, datetime
    from pathlib import Path

    try:
        from omegaconf import OmegaConf

        from roboeval.envs.aloha import ALOHA_TRANSFER_CUBE_ID, make_aloha_env
        from roboeval.envs.success import (
            SuccessCriterion,
            TransferCubeSuccessDetector,
        )
        from roboeval.evaluation.calibration import (
            calibrate_target_xy,
            calibration_to_dict,
        )
        from roboeval.evaluation.loop import evaluate_policy
        from roboeval.policies.factory import load_policy
    except ImportError as exc:
        _LOG.error("Missing dependency: %s", exc)
        return 1

    cfg = OmegaConf.load(config_path)
    # The calibrate command produces the calibration JSON; it cannot read
    # from it. We build a wide-open detector inline whose only job is to
    # be present (calibrate_target_xy only consults the PRIMARY success
    # signal `r.success`, never `r.success_custom` — see
    # roboeval.evaluation.calibration.calibrate_target_xy).
    placeholder_criterion = SuccessCriterion(
        z_threshold_m=0.05,
        xy_tolerance_m=1.0,
        dwell_steps=10_000,
        target_xy=(0.0, 0.0),
    )

    _LOG.info(
        "Loading %s policy %s on device=%s for calibration",
        str(cfg.policy.kind),
        str(cfg.policy.repo_id),
        str(cfg.policy.device),
    )
    try:
        policy = load_policy(
            kind=str(cfg.policy.kind),
            repo_id=str(cfg.policy.repo_id),
            task=str(cfg.env.task),
            device=str(cfg.policy.device),
        )
    except Exception as exc:  # noqa: BLE001 - cli boundary
        _LOG.exception("Failed to load policy: %s", exc)
        return 2

    _LOG.info("Running %d calibration rollouts (single seed, no W&B)", n_rollouts)
    try:
        result = evaluate_policy(
            env_factory=lambda: make_aloha_env(
                task=str(cfg.env.task),
                episode_length=int(cfg.env.episode_length),
            ),
            policy=policy,
            detector_factory=lambda: TransferCubeSuccessDetector(placeholder_criterion),
            seeds=[0],
            n_rollouts_per_seed=n_rollouts,
            max_steps=int(cfg.eval.max_steps),
            policy_id=str(cfg.policy.repo_id),
            env_id=ALOHA_TRANSFER_CUBE_ID,
        )
    except Exception as exc:  # noqa: BLE001 - cli boundary
        _LOG.exception("Calibration rollouts crashed: %s", exc)
        return 4

    try:
        calib = calibrate_target_xy(result.rollouts)
    except ValueError as exc:
        _LOG.error("Calibration failed: %s", exc)
        return 3

    git_sha = _git_sha()
    payload = calibration_to_dict(
        calib,
        git_sha=git_sha,
        timestamp=datetime.now(UTC).isoformat(),
        source_config=config_path,
        policy_id=str(cfg.policy.repo_id),
        env_id=ALOHA_TRANSFER_CUBE_ID,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))

    print(
        f"\n[roboeval] Calibration complete.\n"
        f"  target_xy        = ({calib.target_xy[0]:.5f}, "
        f"{calib.target_xy[1]:.5f})\n"
        f"  xy_tolerance_m   = {calib.xy_tolerance_m:.5f}  "
        f"({calib.percentile:.0f}th percentile of "
        f"||endpoint - centroid||)\n"
        f"  n_successes      = {calib.n_successes} / {calib.n_rollouts}\n"
        f"  written to       = {out}\n"
        f"\nUpdate configs/baseline/act_nominal*.yaml `success.target_xy` and\n"
        f"`success.xy_tolerance_m` with the values above; then re-run the fast\n"
        f"smoke to confirm mean_tsr_custom converges to mean_tsr_native.\n"
    )
    return 0


def _git_sha() -> str:
    """Return the current ``HEAD`` short SHA, or ``'unknown'`` on failure.

    Returns:
        Short git SHA string.
    """
    import subprocess

    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha or "unknown"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


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

    calibrate = subparsers.add_parser(
        "calibrate",
        help="Calibrate target_xy + xy_tolerance from N nominal rollouts.",
    )
    calibrate.add_argument(
        "--config",
        required=True,
        help="YAML config that sources policy/env settings (e.g. act_nominal.yaml).",
    )
    calibrate.add_argument(
        "--output",
        default="data/calibration/transfer_cube_target_xy.json",
        help="Path to write the calibration JSON.",
    )
    calibrate.add_argument(
        "--n-rollouts",
        type=int,
        default=50,
        help=(
            "Number of single-seed rollouts to run (default: 50; see "
            "roboeval/evaluation/calibration.py for the justification)."
        ),
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
    if args.command == "calibrate":
        return _cmd_calibrate(
            config_path=str(args.config),
            output_path=str(args.output),
            n_rollouts=int(args.n_rollouts),
        )
    # Unreachable: subparsers(required=True) enforces a known command.
    raise AssertionError(f"unhandled command: {args.command!r}")
