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


def _build_reward_fn(cfg: Any, target_xy: tuple[float, float]) -> Any:
    """Build a reward_fn closure from the residual config block.

    Reads ``cfg.residual.reward.kind`` (``"sparse"`` or ``"shaped"``)
    and ``cfg.residual.reward.shaping_weight``. Returns a callable
    matching :class:`roboeval.residual.env_wrapper.RewardFn`.
    """
    from roboeval.residual.reward import combined_reward, sparse_success_reward

    reward_cfg = cfg.residual.reward
    kind = str(reward_cfg.kind)
    if kind == "sparse":

        def _reward_sparse(info: Any, _cube_xy: Any) -> float:
            return sparse_success_reward(info)

        return _reward_sparse
    if kind == "shaped":
        shaping_weight = float(reward_cfg.shaping_weight)

        def _reward_shaped(info: Any, cube_xy: Any) -> float:
            return combined_reward(
                info,
                cube_xy,
                target_xy,
                shaping_weight=shaping_weight,
            )

        return _reward_shaped
    raise ValueError(
        f"unknown residual.reward.kind {kind!r}; expected 'sparse' or 'shaped'."
    )


def _cmd_residual_train(config_path: str) -> int:
    """Train a PPO residual against a perturbed env (PRD §8 Conditions B/C).

    Loads the residual config (which extends an eval baseline + adds
    ``residual:`` and ``perturbation:`` blocks), constructs the frozen
    base policy, builds the env factory with the chosen perturbation,
    constructs the reward function from the config, and runs SB3 PPO
    via :func:`roboeval.residual.train.train_residual`.

    Args:
        config_path: Path to the YAML config (e.g.
            ``configs/residual/residual_ppo_y+5cm_sparse.yaml``).

    Returns:
        ``0`` on success, ``1`` on missing dependency, ``2`` on policy
        load failure, ``3`` on training failure.
    """
    try:
        from roboeval.envs.aloha import make_aloha_env
        from roboeval.envs.perturb import make_perturbed_env
        from roboeval.evaluation.calibration import register_calibration_resolver
        from roboeval.evaluation.config import load_eval_config
        from roboeval.policies.factory import load_policy
        from roboeval.residual.policy import ResidualCompositor
        from roboeval.residual.train import train_residual
    except ImportError as exc:
        _LOG.error("Missing dependency: %s", exc)
        _LOG.error("Run `uv pip install -e '.[dev]'` to install the stack.")
        return 1

    register_calibration_resolver()
    cfg = load_eval_config(config_path)

    try:
        base_policy = load_policy(
            kind=str(cfg.policy.kind),
            repo_id=str(cfg.policy.repo_id),
            task=str(cfg.env.task),
            device=str(cfg.policy.device),
        )
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Failed to load base policy: %s", exc)
        return 2

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

    target_xy_list = list(cfg.success.target_xy)
    target_xy = (float(target_xy_list[0]), float(target_xy_list[1]))
    reward_fn = _build_reward_fn(cfg, target_xy)

    compositor = ResidualCompositor(alpha_init=float(cfg.residual.alpha_init))
    res_cfg = cfg.residual

    try:
        save_path = train_residual(
            base_env_factory=_env_factory,
            base_policy=base_policy,
            compositor=compositor,
            reward_fn=reward_fn,
            output_dir=str(res_cfg.output_dir),
            total_timesteps=int(res_cfg.total_timesteps),
            learning_rate=float(res_cfg.learning_rate),
            n_steps=int(res_cfg.n_steps),
            batch_size=int(res_cfg.batch_size),
            n_epochs=int(res_cfg.n_epochs),
            gamma=float(res_cfg.gamma),
            seed=int(res_cfg.seed),
        )
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Residual PPO training crashed: %s", exc)
        return 3

    print(f"\n[roboeval residual train] Done. Saved model to: {save_path}")
    return 0


def _cmd_residual_evaluate(config_path: str, residual_path: str) -> int:
    """Evaluate a trained PPO residual via the standard evaluate_policy pipeline.

    Reuses the eval CLI's machinery (env factory, success detector, W&B
    logging, classifier post-processing) but substitutes
    :class:`ResidualCompositePolicy` for the bare base policy. Produces
    the same ``auto_labels_<run_id>.json`` artifact and W&B summary, so
    Condition B / C TSR can be compared directly against Condition A.

    Args:
        config_path: YAML config (residual training config; this command
            re-reads the perturbation + success blocks from it).
        residual_path: Path to the SB3 PPO ``.zip`` saved by training.
    """
    try:
        from stable_baselines3 import PPO

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
        from roboeval.residual.composite import ResidualCompositePolicy
        from roboeval.residual.policy import ResidualCompositor
        from roboeval.taxonomy import (
            classify_rollout,
            compute_distribution,
            write_auto_labels,
        )
    except ImportError as exc:
        _LOG.error("Missing dependency: %s", exc)
        _LOG.error("Run `uv pip install -e '.[dev]'` to install the stack.")
        return 1

    from datetime import datetime

    register_calibration_resolver()
    cfg = load_eval_config(config_path)
    target_xy_list = list(cfg.success.target_xy)
    criterion = SuccessCriterion(
        z_threshold_m=float(cfg.success.z_threshold_m),
        xy_tolerance_m=float(cfg.success.xy_tolerance_m),
        dwell_steps=int(cfg.success.dwell_steps),
        target_xy=(float(target_xy_list[0]), float(target_xy_list[1])),
    )

    try:
        base_policy = load_policy(
            kind=str(cfg.policy.kind),
            repo_id=str(cfg.policy.repo_id),
            task=str(cfg.env.task),
            device=str(cfg.policy.device),
        )
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Failed to load base policy: %s", exc)
        return 2

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

    compositor = ResidualCompositor(alpha_init=float(cfg.residual.alpha_init))
    residual_model = PPO.load(residual_path)
    composite = ResidualCompositePolicy(
        base_policy=base_policy,
        residual_model=residual_model,
        compositor=compositor,
    )

    reward_kind = str(cfg.residual.reward.kind)
    run_name = (
        f"{cfg.wandb.name_prefix}_eval_" f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    config_dict = {
        "policy_kind": "residual_act",
        "policy_id": composite.policy_id,
        "env_id": ALOHA_TRANSFER_CUBE_ID,
        "device": composite.device,
        "n_rollouts_per_seed": int(cfg.eval.n_rollouts_per_seed),
        "seeds": list(cfg.eval.seeds),
        "max_steps": int(cfg.eval.max_steps),
        "perturbation_kind": perturb_kind or "none",
        "perturbation_params": perturb_params,
        "residual_path": residual_path,
        "residual_reward_kind": reward_kind,
        "residual_alpha_init": float(cfg.residual.alpha_init),
        "lerobot_version": "0.4.4",
    }

    try:
        with wandb_run(
            project=str(cfg.wandb.project),
            name=run_name,
            config=config_dict,
            tags=[*list(cfg.wandb.tags), "evaluate"],
            mode=str(cfg.wandb.mode),
        ) as handle:
            result = evaluate_policy(
                env_factory=_env_factory,
                policy=composite,
                detector_factory=lambda: TransferCubeSuccessDetector(criterion),
                seeds=list(cfg.eval.seeds),
                n_rollouts_per_seed=int(cfg.eval.n_rollouts_per_seed),
                max_steps=int(cfg.eval.max_steps),
                policy_id=composite.policy_id,
                env_id=ALOHA_TRANSFER_CUBE_ID,
                on_rollout=handle.log_rollout,
            )
            handle.log_summary(result)
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
                policy_id=composite.policy_id,
                env_id=ALOHA_TRANSFER_CUBE_ID,
                perturbation_kind=perturb_kind or "none",
                perturbation_params=perturb_params,
                perturbation_applied=perturbation_applied,
            )
            handle.log_distribution(distribution)

            summary = (
                f"\n[roboeval residual evaluate] Done.\n"
                f"  mean_tsr        = {result.mean_tsr:.3f} +/- {result.std_tsr:.3f}\n"
                f"  mean_tsr_custom = "
                f"{result.mean_tsr_custom:.3f} +/- {result.std_tsr_custom:.3f}\n"
                f"  per_seed_tsr    = {result.per_seed_tsr}\n"
                f"  failure_dist    = {distribution}\n"
                f"  auto_labels     = {labels_path}\n"
            )
            print(summary)
            if handle.url:
                print(f"  wandb_run_url   = {handle.url}")
    except Exception as exc:  # noqa: BLE001 - cli-level boundary
        _LOG.exception("Residual evaluation crashed: %s", exc)
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

    residual = subparsers.add_parser(
        "residual",
        help="Phase 4 residual RL: train and evaluate PPO residuals.",
    )
    residual_sub = residual.add_subparsers(dest="residual_cmd", required=True)
    residual_train = residual_sub.add_parser(
        "train",
        help="Train a PPO residual on top of a frozen base policy.",
    )
    residual_train.add_argument(
        "--config",
        required=True,
        help=(
            "Residual training config "
            "(e.g. configs/residual/residual_ppo_y+5cm_sparse.yaml)."
        ),
    )
    residual_eval = residual_sub.add_parser(
        "evaluate",
        help="Evaluate a trained residual via standard evaluate_policy pipeline.",
    )
    residual_eval.add_argument(
        "--config",
        required=True,
        help="Same config used to train (residual + perturbation + success).",
    )
    residual_eval.add_argument(
        "--residual-path",
        required=True,
        help=(
            "Path to the saved PPO model "
            "(the .zip file written by stable_baselines3)."
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
    if args.command == "residual":
        if args.residual_cmd == "train":
            return _cmd_residual_train(config_path=str(args.config))
        if args.residual_cmd == "evaluate":
            return _cmd_residual_evaluate(
                config_path=str(args.config),
                residual_path=str(args.residual_path),
            )
        raise AssertionError(f"unhandled residual_cmd: {args.residual_cmd!r}")
    # Unreachable: subparsers(required=True) enforces a known command.
    raise AssertionError(f"unhandled command: {args.command!r}")
