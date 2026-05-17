"""Reproducibility regression for :func:`run_rollout`.

Motivation
----------
Across Week-4 → Week-5 sessions ``mean_tsr_custom`` on the nominal cell
drifted from 0.680 to 0.727 while ``mean_tsr`` stayed pegged at 0.800
with identical per-seed-group breakdowns. The trajectory aggregates and
the new physics reads (``contact_fn``, ``gripper_xy_fn``) had landed
between those sessions, so the working hypothesis was either:

1. The new ``dm_control`` reads advance the internal mjData state and
   knock the custom-success detector off the trajectory it would
   otherwise have seen, OR
2. MPS / lerobot policy nondeterminism that's always been there but
   only became visible in the custom-TSR column.

This test isolates hypothesis (1) from (2): with a deterministic mock
env, a seeded-RNG policy, and deterministic accessors, two rollouts
with the same ``episode_seed`` must produce bit-identical
:class:`RolloutResult` fields (modulo ``wall_time_s``). If this test
ever regresses, our orchestration code is non-deterministic and the
fix lives in this repo. If this test passes but real-env runs still
drift, the cause is downstream (MPS / lerobot / dm_control internals).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium import spaces

from roboeval.envs.success import SuccessCriterion, TransferCubeSuccessDetector
from roboeval.evaluation.rollout import run_rollout

_TEST_CRITERION = SuccessCriterion(
    z_threshold_m=0.05,
    xy_tolerance_m=0.05,
    dwell_steps=3,
    target_xy=(0.0, 0.0),
)


class _DeterministicMockEnv(gym.Env[Any, Any]):
    """Tiny env that ignores actions and reports a fixed truncation point."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, action_dim: int = 14, max_steps: int = 30):
        self.observation_space = spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=-1, high=1, shape=(action_dim,), dtype=np.float64
                )
            }
        )
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self._max_steps = max_steps
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return {"agent_pos": np.zeros(14, dtype=np.float64)}, {"is_success": False}

    def step(self, action):
        del action
        self._step += 1
        obs = {"agent_pos": np.zeros(14, dtype=np.float64)}
        reward = 0
        terminated = False
        truncated = self._step >= self._max_steps
        return obs, reward, terminated, truncated, {"is_success": False}


class _SeededRandomPolicy:
    """Pulls actions from the global numpy RNG that ``seed_everything`` seeds.

    Same ``episode_seed`` → same global RNG state at policy.reset() time →
    same action sequence. Crucially the policy does NOT carry its own
    Generator; it uses ``np.random.*`` so determinism comes from the
    seed_everything call inside run_rollout.
    """

    policy_id = "seeded_random"
    device = "cpu"

    def __init__(self, action_dim: int = 14, scale: float = 0.5):
        self._dim = action_dim
        self._scale = scale
        self.n_reset_calls = 0

    def reset(self) -> None:
        self.n_reset_calls += 1

    def select_action(
        self, observation: Mapping[str, object]
    ) -> npt.NDArray[np.float32]:
        del observation
        return (np.random.standard_normal(self._dim) * self._scale).astype(np.float32)


def _drifting_cube_state(env: gym.Env[Any, Any]) -> npt.NDArray[np.float64]:
    """Cube xy drifts as a function of the env's internal step counter."""
    step = getattr(env, "_step", 0)
    return np.array(
        [0.001 * step, 0.0005 * step, 0.05, 1.0, 0.0, 0.0, 0.0],
        dtype=np.float64,
    )


def _fixed_gripper_xy(
    env: gym.Env[Any, Any],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Step-dependent but deterministic per-arm xy."""
    step = getattr(env, "_step", 0)
    left = np.array([0.02 + 0.0001 * step, 0.01], dtype=np.float64)
    right = np.array([0.0, 0.0 + 0.0002 * step], dtype=np.float64)
    return left, right


def _step_gated_contact(env: gym.Env[Any, Any]) -> bool:
    """Returns True only on step 5 onward — exercises the OR'd contact bit."""
    return bool(getattr(env, "_step", 0) >= 5)


def _run_one(seed: int) -> Any:
    return run_rollout(
        env=_DeterministicMockEnv(max_steps=30),
        policy=_SeededRandomPolicy(),
        success_detector=TransferCubeSuccessDetector(_TEST_CRITERION),
        seed_group=0,
        rollout_idx=0,
        episode_seed=seed,
        max_steps=30,
        cube_state_fn=_drifting_cube_state,
        gripper_xy_fn=_fixed_gripper_xy,
        contact_fn=_step_gated_contact,
    )


def _fields_to_compare() -> list[str]:
    """Every RolloutResult field except ``wall_time_s`` (wall-clock)."""
    from roboeval.evaluation.types import RolloutResult

    return [
        f.name for f in dataclasses.fields(RolloutResult) if f.name != "wall_time_s"
    ]


def test_rollout_is_deterministic_across_two_same_seed_runs():
    a = _run_one(seed=123)
    b = _run_one(seed=123)
    for field in _fields_to_compare():
        va, vb = getattr(a, field), getattr(b, field)
        assert va == vb, f"field {field!r} diverged: a={va!r} b={vb!r}"


def test_rollout_exercises_all_trajectory_aggregates():
    """Sanity-check that the determinism test isn't trivially zero everywhere.

    A green determinism test on all-zero aggregates would be useless — it'd
    just be asserting that 0.0 == 0.0. This guards against future changes
    that accidentally make the mock setup produce trivial values.
    """
    r = _run_one(seed=123)
    assert (
        r.action_sign_flip_rate > 0.0
    ), "seeded gaussian actions should produce sign flips"
    assert r.contact_made is True, "contact_fn fires from step 5 onward"
    assert (
        r.last_50_step_cube_displacement_m > 0.0
    ), "drifting cube should accumulate xy displacement"
    assert r.terminal_eef_xy_distance_m is not None
    assert r.terminal_eef_xy_distance_m > 0.0


def test_different_seeds_produce_different_trajectory_aggregates():
    """Determinism doesn't mean "all seeds collapse to one answer"."""
    a = _run_one(seed=1)
    b = _run_one(seed=2)
    # At least one trajectory aggregate must differ between seeds — the
    # policy is RNG-driven, so action_sign_flip_rate is the obvious diff.
    assert a.action_sign_flip_rate != b.action_sign_flip_rate
