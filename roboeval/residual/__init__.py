"""Residual RL trainer (Stable-Baselines3 PPO).

Implements the residual RL design from PRD Section 8: a small MLP residual
head trained with PPO on top of a frozen base policy, with the action
composition ``final = base + alpha * residual`` and ablations across sparse
and shaped rewards (PRD Section 8.3).
"""

from __future__ import annotations
