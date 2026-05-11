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
    # Unreachable: subparsers(required=True) enforces a known command.
    raise AssertionError(f"unhandled command: {args.command!r}")
