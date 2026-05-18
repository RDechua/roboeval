"""Residual-policy architectural primitives (PRD §8.2).

Two PyTorch modules:

* :class:`ResidualMLP` — the trainable residual itself: a small
  2-hidden-layer MLP (256 units, GELU) that maps
  ``(obs_features, base_action) -> action_residual``. Architecture
  matches the PRD spec verbatim.
* :class:`ResidualCompositor` — combines the frozen base action with
  the residual: ``a = clamp(a_base + alpha * a_residual, -1, 1)`` where
  ``alpha = sigmoid(alpha_logit)`` is a learnable scalar initialised
  so ``alpha ≈ 0.1`` at the start of training. The sigmoid clipping
  guarantees the residual cannot fully replace the base policy (PRD
  §8.2 design intent).

These primitives are intentionally backend-agnostic: they consume and
produce plain PyTorch tensors, with no knowledge of gym environments,
SB3, or the ALOHA observation schema. Higher-level orchestration lives
in :mod:`roboeval.residual.env_wrapper` and
:mod:`roboeval.residual.train`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

_DEFAULT_HIDDEN_DIM: int = 256
"""PRD §8.2: 'Small MLP, 2 hidden layers x 256 units, GELU activations'."""

_DEFAULT_ALPHA_INIT: float = 0.1
"""PRD §8.2: 'a learnable scalar initialised at 0.1'."""


class ResidualMLP(nn.Module):
    """The trainable residual: MLP from (obs_features, base_action) to action delta.

    Architecture: ``Linear(in_dim, 256) → GELU → Linear(256, 256) → GELU →
    Linear(256, action_dim)``. Output is unbounded; the compositor handles
    the action-space clipping after combining with the base action.

    The forward signature takes obs_features and base_action as separate
    tensors so callers can choose whether to detach base_action from the
    base policy's autograd graph. (For PPO, the base is frozen so detach
    is the right default; the wrapper handles that.)
    """

    def __init__(
        self,
        obs_feature_dim: int,
        action_dim: int = 14,
        hidden_dim: int = _DEFAULT_HIDDEN_DIM,
    ) -> None:
        """Build the residual MLP.

        Args:
            obs_feature_dim: Dimensionality of the per-step observation
                feature vector (e.g. ACT encoder output). 0 is legal
                for unit tests that pass raw actions only.
            action_dim: Action-space dimensionality. ALOHA bimanual is 14.
            hidden_dim: Width of both hidden layers (PRD spec: 256).
        """
        super().__init__()
        if obs_feature_dim < 0:
            raise ValueError(f"obs_feature_dim must be >= 0; got {obs_feature_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0; got {action_dim}")
        input_dim = obs_feature_dim + action_dim
        self._obs_feature_dim = int(obs_feature_dim)
        self._action_dim = int(action_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, action_dim),
        )

    @property
    def obs_feature_dim(self) -> int:
        """Read-only obs feature dimensionality."""
        return self._obs_feature_dim

    @property
    def action_dim(self) -> int:
        """Read-only action dimensionality."""
        return self._action_dim

    def forward(
        self, obs_features: torch.Tensor, base_action: torch.Tensor
    ) -> torch.Tensor:
        """Compute the action residual from (obs_features, base_action).

        Args:
            obs_features: ``(batch, obs_feature_dim)`` or
                ``(obs_feature_dim,)``. May be a zero-width tensor if
                ``obs_feature_dim == 0``.
            base_action: ``(batch, action_dim)`` or ``(action_dim,)``.

        Returns:
            Residual tensor with the same leading shape as the inputs
            and trailing dim ``action_dim``. Unbounded — compositor
            clips after combining.
        """
        x = torch.cat([obs_features, base_action], dim=-1)
        out: torch.Tensor = self.net(x)
        return out


class ResidualCompositor(nn.Module):
    """Combine a frozen base action with a residual delta.

    Implements PRD §8.2: ``a = a_base + alpha * a_residual``, clipped to
    the action space ``[-1, 1]``. ``alpha = sigmoid(alpha_logit)`` is a
    learnable scalar in ``(0, 1)``. The logit is initialised so
    ``alpha ≈ alpha_init`` at start of training (default 0.1).

    The sigmoid clipping ensures alpha < 1, so the residual can never
    fully replace the base — a structural guarantee that the residual
    is a *correction*, not a substitute policy.
    """

    def __init__(
        self,
        alpha_init: float = _DEFAULT_ALPHA_INIT,
        action_min: float = -1.0,
        action_max: float = 1.0,
    ) -> None:
        """Construct the compositor with a learnable mixing scalar.

        Args:
            alpha_init: Initial value of alpha after sigmoid. Must be in
                the open interval (0, 1). Default 0.1 per PRD.
            action_min: Lower clip bound. ALOHA action space is
                ``Box(-1, 1)``; default ``-1.0`` matches.
            action_max: Upper clip bound.

        Raises:
            ValueError: If ``alpha_init`` is outside ``(0, 1)`` or if
                ``action_min >= action_max``.
        """
        super().__init__()
        if not (0.0 < alpha_init < 1.0):
            raise ValueError(f"alpha_init must be in (0, 1); got {alpha_init}")
        if action_min >= action_max:
            raise ValueError(
                f"action_min must be < action_max; " f"got ({action_min}, {action_max})"
            )
        # Store as pre-sigmoid logit so the parameter is unbounded for the
        # optimiser; the constraint alpha ∈ (0, 1) is enforced by sigmoid().
        logit = math.log(alpha_init / (1.0 - alpha_init))
        self.alpha_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))
        self._action_min = float(action_min)
        self._action_max = float(action_max)

    @property
    def alpha(self) -> torch.Tensor:
        """Current sigmoid-clipped mixing scalar (Tensor, shape ``()``)."""
        out: torch.Tensor = torch.sigmoid(self.alpha_logit)
        return out

    def forward(
        self, base_action: torch.Tensor, residual_action: torch.Tensor
    ) -> torch.Tensor:
        """Compose and clip.

        Args:
            base_action: Output of the frozen base policy.
            residual_action: Output of :class:`ResidualMLP`.

        Returns:
            ``clamp(base + alpha * residual, action_min, action_max)``.
        """
        combined = base_action + self.alpha * residual_action
        return torch.clamp(combined, self._action_min, self._action_max)


__all__ = ["ResidualCompositor", "ResidualMLP"]
