"""Policy protocol — the minimal interface the rollout loop depends on.

Any object that implements :class:`Policy` can be passed to
:func:`roboeval.evaluation.rollout.run_rollout`. This lets the v1.0 ACT
adapter, v1.1 Diffusion Policy adapter, and the v1.0 residual-RL policy
(PRD Section 8) all plug into the same harness without code changes
elsewhere.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

ObservationDict = Mapping[str, object]
"""Env-side observation dict; concrete keys are env-specific.

For ALOHA with ``obs_type="pixels_agent_pos"`` the shape is::

    {"pixels": {"top": np.uint8(H, W, 3)}, "agent_pos": np.float64(14,)}

The Protocol stays env-agnostic by typing the values as ``object``;
implementations narrow internally.
"""


@runtime_checkable
class Policy(Protocol):
    """Minimal evaluation contract.

    The rollout loop only ever calls :meth:`select_action` and
    :meth:`reset`; the CLI additionally reads :attr:`policy_id` and
    :attr:`device` for W&B logging. Any object that exposes those four
    things — whether a wrapped pretrained checkpoint, a residual head
    on top of a frozen base, or a hand-coded scripted policy for
    testing — can be evaluated identically.

    Attributes:
        policy_id: Identifier the adapter was constructed from (typically
            the HuggingFace repo id, e.g.
            ``"lerobot/act_aloha_sim_transfer_cube_human"``). Logged
            verbatim to the W&B run config for reproducibility.
        device: Resolved torch device string (``"mps"``, ``"cuda"``, or
            ``"cpu"``). May differ from the device requested at
            construction time if the adapter fell back.
    """

    policy_id: str
    device: str

    def select_action(self, observation: ObservationDict) -> npt.NDArray[np.float32]:
        """Compute the next action for the current observation.

        Args:
            observation: Env-specific observation dict, exactly as returned
                by ``env.reset()`` / ``env.step()``. Concrete schema depends
                on the env's ``observation_space``.

        Returns:
            1-D ``float32`` action array of length equal to the env's
            ``action_space.shape[0]``. The rollout loop passes it
            unchanged to ``env.step(action)``.
        """
        ...

    def reset(self) -> None:
        """Clear internal episode state (action queues, recurrent state, ...).

        Called exactly once at the start of every rollout, before
        :meth:`select_action`.
        """
        ...
