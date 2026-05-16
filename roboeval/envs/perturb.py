"""Perturbation env wrappers for the PRD §6.4 robustness suite.

v1.0 ships only the **spatial** axis (cube initial-position shift).
The other three axes (visual lighting + distractor; dynamic mid-rollout
push; temporal action-delay) have placeholder factory branches that
raise :class:`NotImplementedError` so a config requesting them fails
loudly at load time rather than silently running nominal.

Each axis follows the same contract: a ``gym.Wrapper`` that the
``env_factory`` callable in :mod:`roboeval.cli` composes onto a freshly
built ALOHA env. The wrappers do not change the rollout loop, the
success detector, or the policy adapter — perturbations are an env-side
concern, by design (PRD §5.1 architecture principle: each component
swappable from a single config flag).

The :func:`make_perturbed_env` factory is the single entry point the
CLI uses; ``perturbation:`` blocks in configs name the kind and supply
its kwargs.
"""

from __future__ import annotations

from typing import Any, Literal, get_args

import gymnasium as gym

# Cube qpos slice — same convention as :mod:`roboeval.envs.aloha`. The cube's
# 7-element free-joint state occupies the LAST 7 entries of physics.data.qpos
# (xyz + quat). The shift is applied to qpos[16] and qpos[17] (cube x, y).
_CUBE_X_INDEX: int = 16
_CUBE_Y_INDEX: int = 17

PerturbationKind = Literal["spatial", "visual", "dynamic", "temporal"]
"""Supported perturbation axes per PRD §6.4."""

_SUPPORTED_KINDS: frozenset[str] = frozenset(get_args(PerturbationKind))


class SpatialShiftWrapper(gym.Wrapper[Any, Any, Any, Any]):
    """Shift the cube's initial xy position by a fixed delta at reset time.

    Implements the PRD §6.4 spatial axis. The shift is **deterministic**
    (same delta every reset) so reproducibility holds across runs at the
    same seed — the perturbation contributes a known systematic bias,
    not noise.

    The wrapper writes ``physics.data.qpos[16] += dx_m`` and
    ``physics.data.qpos[17] += dy_m`` immediately after the underlying
    env's reset, then calls ``physics.forward()`` to recompute
    kinematics so any subsequent ``get_cube_state`` call reflects the
    perturbed pose. The observation returned by ``reset()`` is the one
    the underlying env produced **before** the shift; if that proves to
    mislead the policy's first action attribution, refresh it via the
    underlying task's ``get_observation`` (TODO when validated on M1).
    """

    def __init__(self, env: gym.Env[Any, Any], dx_m: float, dy_m: float) -> None:
        """Construct a spatial-shift wrapper.

        Args:
            env: An ALOHA Transfer Cube env (typically from
                :func:`roboeval.envs.aloha.make_aloha_env`).
            dx_m: Cube x-shift in metres. Sign convention matches the
                env's world frame.
            dy_m: Cube y-shift in metres.
        """
        super().__init__(env)
        self._dx_m = float(dx_m)
        self._dy_m = float(dy_m)

    @property
    def dx_m(self) -> float:
        """Read-only x-shift in metres (for logging/diagnostics)."""
        return self._dx_m

    @property
    def dy_m(self) -> float:
        """Read-only y-shift in metres."""
        return self._dy_m

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the underlying env, then apply the deterministic xy shift."""
        obs, info = self.env.reset(seed=seed, options=options)
        physics = self.env.unwrapped._env.physics  # type: ignore[attr-defined]
        physics.data.qpos[_CUBE_X_INDEX] += self._dx_m
        physics.data.qpos[_CUBE_Y_INDEX] += self._dy_m
        # Recompute kinematics so the perturbed pose propagates to any
        # subsequent get_cube_state() / camera-render call without
        # advancing simulation time.
        physics.forward()
        return obs, info


def _make_visual_wrapper(env: gym.Env[Any, Any], **params: Any) -> gym.Env[Any, Any]:
    """Placeholder for the visual perturbation axis (PRD §6.4)."""
    del env, params
    raise NotImplementedError(
        "visual perturbation (lighting ±30/60%, distractor) lands later in "
        "Week 4. Use kind='spatial' for now."
    )


def _make_dynamic_wrapper(env: gym.Env[Any, Any], **params: Any) -> gym.Env[Any, Any]:
    """Placeholder for the dynamic perturbation axis (PRD §6.4)."""
    del env, params
    raise NotImplementedError(
        "dynamic perturbation (mid-rollout cube push at 25/50/75% of "
        "nominal completion) lands later in Week 4. Requires a "
        "perturbation_callback hook in run_rollout."
    )


def _make_temporal_wrapper(env: gym.Env[Any, Any], **params: Any) -> gym.Env[Any, Any]:
    """Placeholder for the temporal perturbation axis (PRD §6.4)."""
    del env, params
    raise NotImplementedError(
        "temporal perturbation (action delay 1/3/5 steps) lands later in "
        "Week 4. Trivial: a collections.deque in front of env.step."
    )


def make_perturbed_env(
    env: gym.Env[Any, Any],
    kind: str,
    **params: Any,
) -> gym.Env[Any, Any]:
    """Wrap ``env`` with the perturbation wrapper named by ``kind``.

    Args:
        env: A bare ALOHA env to wrap.
        kind: One of :data:`PerturbationKind`. Only ``"spatial"`` is
            implemented in v1.0; the others raise
            :class:`NotImplementedError` with a pointer to the relevant
            Week-4 milestone.
        **params: Per-kind keyword arguments. For ``kind="spatial"``,
            requires ``dx_m: float`` and ``dy_m: float`` (either may be
            zero). Extra keys are forwarded to the wrapper unchanged;
            unknown keys raise ``TypeError`` from the wrapper itself.

    Returns:
        A ``gym.Env`` with the perturbation applied.

    Raises:
        ValueError: If ``kind`` is not in :data:`_SUPPORTED_KINDS`.
        NotImplementedError: If ``kind`` is reserved but the adapter
            has not yet been implemented (visual / dynamic / temporal).
    """
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"unknown perturbation kind {kind!r}; "
            f"supported: {sorted(_SUPPORTED_KINDS)}"
        )
    if kind == "spatial":
        return SpatialShiftWrapper(
            env,
            dx_m=float(params.get("dx_m", 0.0)),
            dy_m=float(params.get("dy_m", 0.0)),
        )
    if kind == "visual":
        return _make_visual_wrapper(env, **params)
    if kind == "dynamic":
        return _make_dynamic_wrapper(env, **params)
    if kind == "temporal":
        return _make_temporal_wrapper(env, **params)
    raise AssertionError(f"unhandled supported kind: {kind!r}")


def _cube_xy_indices() -> tuple[int, int]:
    """Return the (x, y) qpos indices the SpatialShiftWrapper writes to.

    Exported so tests can target the same indices the wrapper writes to
    when building synthetic qpos arrays.
    """
    return (_CUBE_X_INDEX, _CUBE_Y_INDEX)


__all__ = [
    "PerturbationKind",
    "SpatialShiftWrapper",
    "make_perturbed_env",
]
