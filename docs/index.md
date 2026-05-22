# RoboEval

Failure-mode study and residual RL for open-source robot-learning policies on the bimanual ALOHA Transfer Cube task.

[![Live Demo](https://img.shields.io/badge/HF%20Spaces-Live%20Demo-blue?logo=huggingface)](https://huggingface.co/spaces/rubenodechua/roboeval)
[![Writeup](https://img.shields.io/badge/Blog-Honest%20null-blueviolet)](blog/2026-05-21-honest-null-residual.md)
[![Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/RDechua/roboeval)

## What this is

RoboEval is a typed Python codebase plus a public-facing research artifact. It measures where ACT
breaks under realistic perturbation (spatial cube shifts, action delays), classifies failure modes
into six operational categories, and ships a residual RL ablation that attempts to recover the
top-frequency failure — with an honest null result and a documented v1.1 fix path.

## Where to go

- **[Live dashboard](https://huggingface.co/spaces/rubenodechua/roboeval)** — interactive degradation curves + Phase 4 ablation.
- **[Blog post](blog/2026-05-21-honest-null-residual.md)** — the honest-null writeup (~2000 words).
- **[Getting started](getting-started.md)** — `git clone` to first rollout in five commands.
- **[API Reference](reference/index.md)** — auto-generated from the typed `roboeval/` package.
- **[Product spec](PRD.md)** — the requirements doc the whole project ships against.

## Stack

Python 3.11 · LeRobot · MuJoCo · Stable-Baselines3 · Hydra · Weights & Biases · Plotly/Dash · MkDocs.
M1 MPS for inference and PPO training — no CUDA required.

## License

MIT.
