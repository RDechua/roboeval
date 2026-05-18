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

The class doesn't depend on Stable-Baselines3 at import time; SB3 is
only touched inside :meth:`select_action` via the ``residual_model``
parameter's ``predict()`` method.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import torch

from roboeval.policies.base import Policy
from roboeval.residual.policy import ResidualCompositor


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
            deterministic: Whether to ask the residual model for a
                deterministic action (mean of the distribution) or a
                sample. Eval defaults to ``True`` so the reported TSR
                isn't inflated by lucky stochastic samples.
        """
        self._base = base_policy
        self._residual = residual_model
        self._compositor = compositor
        self._deterministic = deterministic
        self.policy_id: str = f"residual({base_policy.policy_id})"
        self.device: str = base_policy.device

    def reset(self) -> None:
        """Forward to the base policy; the residual is stateless across episodes."""
        self._base.reset()

    def select_action(self, observation: Mapping[str, Any]) -> npt.NDArray[np.float32]:
        """Compose base + alpha * residual and return as numpy float32."""
        base_action_np = self._base.select_action(observation)
        residual_action_np, _state = self._residual.predict(
            observation, deterministic=self._deterministic
        )
        with torch.no_grad():
            base_t = torch.from_numpy(np.asarray(base_action_np, dtype=np.float32))
            res_t = torch.from_numpy(np.asarray(residual_action_np, dtype=np.float32))
            composed_t = self._compositor(base_t, res_t)
        out: npt.NDArray[np.float32] = composed_t.cpu().numpy().astype(np.float32)
        return out


__all__ = ["ResidualCompositePolicy"]
