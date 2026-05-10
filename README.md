# RoboEval

> Open-source evaluation harness & failure-mode study for robot learning policies (ACT, Diffusion Policy) with residual RL fine-tuning.

## Overview

RoboEval systematically measures where state-of-the-art imitation learning policies break,
classifies failure modes into an operationalised taxonomy, and demonstrates a residual RL
loop that recovers the highest-frequency failure.

**Stack:** Python · MuJoCo · LeRobot · Stable-Baselines3 · Hydra · Weights & Biases · Plotly  
**Hardware target:** Apple M1 (CPU/MPS) — no CUDA required for evaluation

## Status

🚧 Under active development — v1.0 targeting July 2026

## Structure (coming soon)

- `roboeval/` — core evaluation library
- `configs/` — Hydra experiment configs
- `analysis/` — dashboard and notebooks
- `docs/` — MkDocs documentation

## Getting Started

_Setup instructions coming in v0.1_

## License

MIT
