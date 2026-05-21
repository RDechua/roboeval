"""CLI regression test for ``roboeval evaluate``.

This test guards against breakage of the CLI plumbing (arg parsing, YAML
schema loading, W&B context-manager API, exit-code paths, the
``Policy`` protocol implementation contract) without requiring lerobot,
mujoco, gym-aloha, torch, or the ~80 MB ACT checkpoint download. The
real policy and env are monkeypatched out at the layer where the CLI
imports them — :mod:`roboeval.policies.act_loader` and
:mod:`roboeval.envs.aloha`. Because :mod:`roboeval.cli` uses lazy
imports inside ``_cmd_evaluate``, those names are not module-level
attributes on ``roboeval.cli`` itself, so monkeypatching the source
modules directly is the only thing that takes effect.

The test runs in CI (no ``slow`` marker) and assumes ``numpy``,
``gymnasium``, ``torch``, ``omegaconf``, ``wandb`` are installed (see
``.github/workflows/ci.yml``).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
import pytest
import yaml
from gymnasium import spaces


class _MockAlohaLikeEnv(gym.Env[Any, Any]):
    """Single-env gym that mirrors AlohaTransferCube's observation shape.

    Reports ``info["is_success"]=True`` after ``success_after_n`` steps so
    the rollout loop records a primary success without needing real
    dm_control physics or the cube-state accessor (which is also patched).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, success_after_n: int = 5, max_steps: int = 10) -> None:
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Dict(
                    {
                        "top": spaces.Box(
                            low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                        )
                    }
                ),
                "agent_pos": spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float64),
            }
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(14,), dtype=np.float32)
        self._success_after_n = success_after_n
        self._max_steps = max_steps
        self._step = 0

    def _obs(self) -> dict[str, Any]:
        return {
            "pixels": {"top": np.zeros((480, 640, 3), dtype=np.uint8)},
            "agent_pos": np.zeros(14, dtype=np.float64),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return self._obs(), {"is_success": False}

    def step(self, action):
        del action
        self._step += 1
        terminated = self._step >= self._success_after_n
        truncated = self._step >= self._max_steps and not terminated
        return (
            self._obs(),
            4 if terminated else 0,
            terminated,
            truncated,
            {"is_success": terminated},
        )


class _MockAdapter:
    """Implements :class:`Policy` with zero-action behavior."""

    policy_id = "mock/act_aloha_sim_transfer_cube_human"
    device = "cpu"

    def reset(self) -> None:
        pass

    def select_action(
        self, observation: Mapping[str, object]
    ) -> npt.NDArray[np.float32]:
        del observation
        return np.zeros(14, dtype=np.float32)


def _mock_load_act_policy(
    repo_id: str,
    task: str = "AlohaTransferCube-v0",
    device: str = "mps",
    dataset_repo_id: str | None = None,
) -> _MockAdapter:
    del repo_id, task, device, dataset_repo_id
    return _MockAdapter()


def _mock_make_aloha_env(
    task: str = "AlohaTransferCube-v0",
    episode_length: int = 400,
    obs_type: str = "pixels_agent_pos",
) -> _MockAlohaLikeEnv:
    del task, episode_length, obs_type
    return _MockAlohaLikeEnv(success_after_n=5, max_steps=10)


def _fake_cube_state(env: gym.Env[Any, Any]) -> npt.NDArray[np.float64]:
    del env
    return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@pytest.fixture
def cli_test_config(tmp_path: Path) -> Path:
    """Write a YAML config matching the schema OmegaConf.load consumes."""
    cfg = {
        "policy": {"kind": "act", "repo_id": "mock-act", "device": "cpu"},
        "env": {"task": "AlohaTransferCube-v0", "episode_length": 10},
        "eval": {"seeds": [0], "n_rollouts_per_seed": 2, "max_steps": 10},
        "success": {
            "z_threshold_m": 0.05,
            "xy_tolerance_m": 0.05,
            "dwell_steps": 5,
            "target_xy": [0.0, 0.0],
        },
        "wandb": {
            "project": "roboeval-test",
            "name_prefix": "cli_regression",
            "tags": ["cli-test"],
            "mode": "disabled",
        },
    }
    path = tmp_path / "cli_test.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_evaluate_subcommand_completes_with_mocks(
    monkeypatch: pytest.MonkeyPatch,
    cli_test_config: Path,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`roboeval evaluate` runs end-to-end against mocked policy + env."""
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_MODE", "disabled")

    # Patch at the source modules so cli.py's lazy imports inside
    # _cmd_evaluate pick up the mocks.
    monkeypatch.setattr(
        "roboeval.policies.act_loader.load_act_policy", _mock_load_act_policy
    )
    monkeypatch.setattr("roboeval.envs.aloha.make_aloha_env", _mock_make_aloha_env)
    # evaluate_policy resolves get_cube_state at call time (loop.py change in
    # Week 2.5) — patch it on the loop module's namespace.
    monkeypatch.setattr("roboeval.evaluation.loop.get_cube_state", _fake_cube_state)

    from roboeval.cli import main

    exit_code = main(["evaluate", "--config", str(cli_test_config)])

    assert exit_code == 0, "evaluate subcommand should exit 0 with valid mocks"
    captured = capsys.readouterr()
    assert "Evaluation complete" in captured.out
    assert "n_rollouts      = 2" in captured.out
    assert "mean_tsr" in captured.out
    # Persisted eval_results JSON should exist alongside the mocked W&B run.
    assert "eval_results    =" in captured.out
    assert (tmp_path / "outputs" / "eval" / "cli_regression").exists() or any(
        Path("outputs/eval/cli_regression").glob("eval_results_*.json")
    )


def test_mock_adapter_satisfies_policy_protocol() -> None:
    """The mock used in the CLI test must implement the Policy protocol."""
    from roboeval.policies.base import Policy

    adapter = _MockAdapter()
    assert isinstance(adapter, Policy)


def test_residual_aggregate_subcommand(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`roboeval residual aggregate` reads JSON inputs, prints + persists report."""
    import json

    from roboeval.cli import main

    def _payload(condition: str, seed: int, mean_tsr_custom: float) -> dict[str, Any]:
        base: dict[str, Any] = {
            "schema_version": 1,
            "run_id": f"{condition.lower()}_seed{seed}",
            "perturbation_kind": "spatial",
            "perturbation_params": {"dx_m": 0.0, "dy_m": 0.05},
            "metrics": {
                "mean_tsr_custom": mean_tsr_custom,
                "n_rollouts": 150,
                "n_seed_groups": 3,
            },
        }
        if condition == "A":
            base["policy_kind"] = "act"
        else:
            base["policy_kind"] = "residual_act"
            base["residual"] = {
                "reward_kind": "sparse" if condition == "B" else "shaped",
                "alpha_init": 0.05,
                "log_std_init": -2.0,
            }
        return base

    inputs: list[str] = []
    for cond, means in (
        ("A", [0.30, 0.32, 0.34]),
        ("B", [0.55, 0.60, 0.50]),
        ("C", [0.65, 0.70, 0.60]),
    ):
        for i, m in enumerate(means):
            p = tmp_path / f"eval_results_{cond.lower()}_{i}.json"
            p.write_text(json.dumps(_payload(cond, i, m)))
            inputs.append(str(p))

    out_json = tmp_path / "report.json"
    exit_code = main(["residual", "aggregate", *inputs, "--output", str(out_json)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Phase 4 Ablation" in captured.out
    assert "Delta TSR vs Condition A" in captured.out
    assert "Frozen base only" in captured.out
    assert out_json.exists()
    payload = json.loads(out_json.read_text())
    assert payload["schema_version"] == 2
    assert len(payload["conditions"]) == 3
    assert len(payload["comparisons"]) == 2


def test_residual_aggregate_missing_file_errors(tmp_path: Path) -> None:
    """Aggregator exits non-zero (1) when a passed path doesn't exist."""
    from roboeval.cli import main

    exit_code = main(["residual", "aggregate", str(tmp_path / "does_not_exist.json")])
    assert exit_code == 1


def test_dashboard_build_subcommand_zero_exit_on_valid_data(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`roboeval dashboard build` validates data loaders and exits 0."""
    from roboeval.cli import main

    exit_code = main(["dashboard", "build"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "10" in out


def test_dashboard_run_subcommand_dry_run_zero_exit() -> None:
    """`dashboard run --dry-run` constructs the app but does not start the server."""
    from roboeval.cli import main

    exit_code = main(["dashboard", "run", "--dry-run"])
    assert exit_code == 0
