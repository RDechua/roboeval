# RoboEval

[![Live Demo](https://img.shields.io/badge/HF%20Spaces-Live%20Demo-blue?logo=huggingface)](https://huggingface.co/spaces/RubenoDechua/roboeval)

> Open-source evaluation harness & failure-mode study for robot learning policies (ACT, Diffusion Policy) with residual RL fine-tuning.

## Overview

RoboEval systematically measures where state-of-the-art imitation learning policies break,
classifies failure modes into an operationalised taxonomy, and demonstrates a residual RL
loop that recovers the highest-frequency failure.

**Stack:** Python 3.11 · MuJoCo · LeRobot · Stable-Baselines3 · Hydra · Weights & Biases · Plotly
**Hardware target:** Apple M1 (CPU/MPS) — no CUDA required for evaluation

## Status

Under active development — v1.0 targeting July 2026. Authoritative spec: [`docs/PRD.md`](docs/PRD.md).

## Repository Layout

Follows PRD Section 5.2:

```
roboeval/
├── configs/              # Hydra YAML configs per experiment
│   ├── baseline/
│   ├── perturbation/
│   └── residual_rl/
├── roboeval/             # Core typed Python library
│   ├── envs/             # Gymnasium env wrappers
│   ├── policies/         # LeRobot policy loaders
│   ├── evaluation/       # Rollout engine + metrics
│   ├── taxonomy/         # Failure-mode classifier
│   └── residual/         # SB3 PPO residual trainer
├── analysis/             # Notebooks + Plotly dashboard
├── docs/                 # PRD + MkDocs source
├── tests/                # pytest unit + integration tests
└── .github/workflows/    # CI: ruff + mypy + pytest
```

## Quick Start (Day 1)

```bash
# 1. Create venv and install (Python 3.11)
uv venv .venv && source .venv/bin/activate
uv pip install -e '.[dev]'

# 2. Verify M1 MPS is available
python -c "import torch; print('mps:', torch.backends.mps.is_available())"

# 3. Verify LeRobot 0.4.4 is importable with the new namespace
python -c "from lerobot.policies.act.modeling_act import ACTPolicy; print('ACT loaded')"

# 4. Run the Week 1 smoke rollout (random actions, 10 steps)
roboeval smoke --steps 10

# 5. Install pre-commit hooks
pre-commit install
```

Note: LeRobot 0.4.x removed the `lerobot.common` namespace — import policies directly from `lerobot.policies.*`.

## Quality Gates

```bash
ruff check .
ruff format --check .
mypy --strict roboeval
pytest -q
```

CI runs all four on every push to `main` and every PR.

## License

MIT
