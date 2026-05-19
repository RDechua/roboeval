"""Policy adapter for the trained residual on top of a frozen base.

After PPO training completes, the residual is a Stable-Baselines3 model
(``stable_baselines3.PPO`` saved to disk). To run the standard
:func:`roboeval.evaluation.loop.evaluate_policy` pipeline against the
trained residual — required for the PRD §8.3 ablation table — we need
something that *quacks* like a :class:`roboeval.policies.base.Policy`:
``policy_id``, ``device``, ``reset()``, ``select_action(obs)``.

:class:`ResidualCompositePolicy` is that adapter. On each
``select_action`` call it asks the frozen base for its action, asks the
SB3 model for the residual, composes them through the same
:class:`ResidualCompositor` used during training, and returns the
composed action. The composition is identical to the train-time path,
so eval-time and train-time actions for the same ``(obs, base_action,
residual)`` agree bit-for-bit.

**Observation-format invariant.** PPO trained on the flat Box obs
produced by :class:`ResidualEnvWrapper` (agent_pos + cube_state +
base_action + features). At eval time we must rebuild the same flat
obs before calling ``residual_model.predict``; passing the raw
gym-aloha Dict crashes SB3's ``obs_to_tensor`` with a Box/Dict
mismatch. The ``obs_builder`` parameter is the hook that supplies
that conversion — :func:`roboeval.residual.env_wrapper.build_flat_obs`
in production, ``None`` in tests that mock the SB3 model and don't
inspect the obs.

The class doesn't depend on Stable-Baselines3 at import time; SB3 is
only touched inside :meth:`select_action` via the ``residual_model``
parameter's ``predict()`` method.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch

from roboeval.policies.base import Policy
from roboeval.residual.policy import ResidualCompositor

ObsBuilder = Callable[
    [Mapping[str, Any], npt.NDArray[np.float32]], npt.NDArray[np.float32]
]
"""``obs_builder(obs_dict, base_action) -> flat_obs`` for the SB3 residual."""


@runtime_checkable
class _PredictsActions(Protocol):
    """Minimal interface SB3 ``BaseAlgorithm`` satisfies (and tests can mock)."""

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[npt.NDArray[Any], Any]:
        """SB3's predict signature; returns ``(action, hidden_state)``."""
        ...


class ResidualCompositePolicy:
    """Adapter exposing ``base + alpha * residual`` as a single :class:`Policy`.

    Implements the :class:`roboeval.policies.base.Policy` Protocol
    structurally (no inheritance — duck typing matches the rest of the
    codebase). Used by :func:`roboeval.evaluation.loop.evaluate_policy`
    to score Phase 4 Conditions B and C against Condition A.
    """

    def __init__(
        self,
        base_policy: Policy,
        residual_model: _PredictsActions,
        compositor: ResidualCompositor,
        *,
        obs_builder: ObsBuilder | None = None,
        deterministic: bool = True,
    ) -> None:
        """Wire a frozen base, a trained residual, and the compositor.

        Args:
            base_policy: Frozen base policy (e.g. ACT). Its ``reset()``
                is called when this composite resets.
            residual_model: Anything with SB3's ``predict()`` signature.
                Production usage: ``stable_baselines3.PPO.load(path)``.
                Tests: a mock returning fixed residuals.
            compositor: The same :class:`ResidualCompositor` used at
                training time. Alpha is read from this object; if the
                trained run swept a non-default alpha_init, the eval
                must use the matching value.
            obs_builder: Callable that maps ``(obs_dict, base_action)``
                to the flat Box obs the SB3 residual was trained on.
                When ``None``, the raw ``observation`` is passed through
                to ``residual_model.predict`` unchanged — useful for
                unit tests that don't care about obs shape, but **never
                correct for a real SB3 model** (will crash in
                ``obs_to_tensor``). Production callers must pass
                :func:`roboeval.residual.env_wrapper.build_flat_obs`
                bound to the eval env + same feature_extractor used
                at training.
            deterministic: Whether to ask the residual model for a
                deterministic action (mean of the distribution) or a
                sample. Eval defaults to ``True`` so the reported TSR
                isn't inflated by lucky stochastic samples.
        """
        self._base = base_policy
        self._residual = residual_model
        self._compositor = compositor
        self._obs_builder = obs_builder
        self._deterministic = deterministic
        self.policy_id: str = f"residual({base_policy.policy_id})"
        self.device: str = base_policy.device

    def reset(self) -> None:
        """Forward to the base policy; the residual is stateless across episodes."""
        self._base.reset()

    def select_action(self, observation: Mapping[str, Any]) -> npt.NDArray[np.float32]:
        """Compose base + alpha * residual and return as numpy float32.

        When ``obs_builder`` is set (production path), the raw Dict obs
        is converted to the same flat Box obs PPO trained on before
        being passed to ``residual_model.predict``.
        """
        base_action_np = np.asarray(
            self._base.select_action(observation), dtype=np.float32
        )
        residual_input: Any
        if self._obs_builder is not None:
            residual_input = self._obs_builder(observation, base_action_np)
        else:
            residual_input = observation
        residual_action_np, _state = self._residual.predict(
            residual_input, deterministic=self._deterministic
        )
        with torch.no_grad():
            base_t = torch.from_numpy(base_action_np)
            res_t = torch.from_numpy(np.asarray(residual_action_np, dtype=np.float32))
            composed_t = self._compositor(base_t, res_t)
        out: npt.NDArray[np.float32] = composed_t.cpu().numpy().astype(np.float32)
        return out


__all__ = ["ObsBuilder", "ResidualCompositePolicy"]
