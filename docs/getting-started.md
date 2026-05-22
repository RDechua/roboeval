# Getting started

Five commands to a working rollout on a fresh clone (Python 3.11, macOS M1 or Linux x86).

## 1. Clone

```bash
git clone https://github.com/RDechua/roboeval.git
cd roboeval
```

## 2. Create the venv and install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

(If you don't have `uv`, `python3.11 -m venv .venv && pip install -e '.[dev]'` works too — `uv` is just faster.)

## 3. Verify the stack

```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
python -c "from lerobot.policies.act.modeling_act import ACTPolicy; print('ACT loaded')"
```

Expected: `mps: True` on M1 (or `False` on Linux — both are fine, `torch` falls back to CPU). `ACT loaded` confirms the LeRobot 0.4.x namespace works.

## 4. First smoke rollout

```bash
roboeval smoke --steps 10
```

This runs 10 random-action steps against `gym_aloha/AlohaTransferCube-v0` and prints a per-step trace. If it completes without raising, the dependency stack is healthy.

## 5. Real evaluation (slower — about 50 minutes on M1)

```bash
roboeval evaluate --config configs/baseline/act_nominal.yaml
```

This loads ACT, runs 3 seeds × 50 rollouts on the nominal cell, classifies the rollouts via the
PRD §7.2 taxonomy, and writes a schema-v1 `eval_results_<run_id>.json` plus an `auto_labels_<run_id>.json`
to `outputs/eval/act_nominal/` and `data/taxonomy/` respectively.

## What's next

- **Train a residual** with `roboeval residual train --config configs/residual/residual_ppo_y+5cm_sparse.yaml`.
- **Launch the dashboard locally** with `roboeval dashboard run` — opens http://localhost:8050.
- **Read the writeup** at [blog/2026-05-21-honest-null-residual.md](blog/2026-05-21-honest-null-residual.md).

## Quality gates

The repo's CI runs four gates on every push to `main` and every PR:

```bash
ruff check .
ruff format --check .
mypy --strict roboeval
pytest -q
```

Each should exit 0. If any complains on your machine, open an issue — the gates pass cleanly on
the maintained branches.
