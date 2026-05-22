# Residual RL

`ResidualMLP` (2×256 GELU) + `ResidualCompositor`, sparse / shaped / combined rewards, the gym
wrapper that composes the frozen base with the residual, the SB3 PPO training loop, and the Phase
4 ablation aggregator (Welch's t + bootstrap CI, stdlib-only).

::: roboeval.residual
