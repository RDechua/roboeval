# RoboEval

*Failure-mode study + residual RL for ACT on AlohaTransferCube — an honest-null result, reproducible on a laptop.*

[![CI](https://github.com/RDechua/roboeval/actions/workflows/ci.yml/badge.svg)](https://github.com/RDechua/roboeval/actions/workflows/ci.yml)
[![Docs](https://github.com/RDechua/roboeval/actions/workflows/docs.yml/badge.svg)](https://rdechua.github.io/roboeval/)
[![Live demo](https://img.shields.io/badge/Live%20demo-HF%20Spaces-blue?logo=huggingface&logoColor=white)](https://rubenodechua-roboeval.hf.space/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](pyproject.toml)

I spent a month measuring how a state-of-the-art imitation-learning policy (ACT) breaks
when the world shifts, classified the failures into an operational taxonomy, then built a
residual RL loop to repair the dominant failure. The repair made the policy **worse** —
task success fell 13.3 points. RoboEval is the harness, the data, and the honest-null
diagnosis of why it happened. Everything reproduces on a single Apple Silicon M1, no GPU.

![Cross-axis degradation curve](docs/figures/cross_axis_degradation.png)

*Figure 1 — Cross-axis degradation curve. Spatial perturbation (left) is brittle; temporal
delay (right) is robust. n = 150 rollouts per cell.*

## Headline result

ACT is **brittle to spatial shift and robust to temporal delay**. Moving the cube's start
position costs up to 67 pp of task-success rate (TSR) at −5 cm and 49 pp at +5 cm, while a
5-step action delay costs only 11 pp (n = 150 per cell). On the +5 cm cell — the cleanest
residual-RL target — a PPO residual on the frozen base hurt under both reward shapings:

| Arm | Mean TSR | Δ vs base | p (decrease) |
|---|---|---|---|
| A — frozen base | 0.320 ± 0.059 | — | — |
| B — residual, sparse | 0.187 ± 0.025 | −13.3 pp | ≈ 0.03 |
| C — residual, shaped | 0.213 ± 0.050 | −10.7 pp | ≈ 0.06 |

Mean ± std across 3 seed groups × 50 rollouts; one-sided Welch's t-test for a significant
decrease. The per-seed spread is tight (0.025, sparse) — signal, not noise. Full analysis
in the [long-form writeup](docs/blog/2026-05-21-honest-null-residual.md).

## Live

- **Dashboard** (interactive degradation curves + the ablation): <https://rubenodechua-roboeval.hf.space/>
- **Docs** (typed API reference): <https://rdechua.github.io/roboeval/>
- **Long-form writeup** (~2k words, honest-null): [docs/blog/2026-05-21-honest-null-residual.md](docs/blog/2026-05-21-honest-null-residual.md)

## Quickstart

Requires Python 3.11.

```bash
git clone https://github.com/RDechua/roboeval
cd roboeval
pip install -e .

roboeval smoke --steps 10                                    # validate the dependency stack
roboeval evaluate --config configs/baseline/act_nominal.yaml  # full ACT eval (3 seeds × 50)
```

Other subcommands: `roboeval calibrate --config <path>`, `roboeval residual {train,evaluate,aggregate}`,
`roboeval dashboard {build,run}` (local Dash app on `:8050`).

## What shipped

- **Evaluation harness** — config-driven rollouts over any LeRobot policy, with native and calibrated-geometric success signals.
- **Failure-mode taxonomy** — six operationally-defined categories plus a rule-based classifier emitting per-rollout labels.
- **Residual RL loop** — frozen ACT base + SB3 PPO residual (sparse and shaped reward arms) + a stdlib-only Welch's t aggregator.
- **Interactive dashboard** — Plotly/Dash over 10 perturbation cells + the Phase-4 ablation, deployed on Hugging Face Spaces.
- **Long-form writeup** — a ~2,000-word honest-null blog post.
- **MkDocs site** — typed API reference, auto-deployed to GitHub Pages.
- **Engineering bar** — 285 tests; `ruff`, `ruff format`, `mypy --strict`, and `pytest` all green in CI.

## Repository layout

```
roboeval/
├── roboeval/          # typed library: envs · policies · evaluation · taxonomy · residual · dashboard
├── configs/           # Hydra YAML configs: baseline · perturbation · residual
├── scripts/           # figure, relabel, and headline-build scripts
├── analysis/          # Plotly/Dash app (deploys to HF Spaces)
├── data/              # tracked artifacts: calibration · taxonomy · headline.json
├── docs/              # PRD, research log, figures, MkDocs source
├── tests/             # pytest unit + integration tests
└── .github/workflows/ # CI (ruff · mypy · pytest) + docs deploy
```

## Reproduce the headline numbers

```bash
roboeval evaluate --config configs/baseline/act_nominal.yaml
```

reproduces the nominal-cell TSR (0.80 ± 0.057, 3 seeds × 50). Runs on a single Apple
Silicon M1; **no GPU required** — the Phase-4 residual ablation was ~16 h of on-device PPO
training. The dashboard's runtime data is one tracked file (`data/headline.json`), rebuilt
by `scripts/build_headline_json.py`.

## Status & scope cuts

Phase 4 is closed with an honest null (the table above); Phase 5 (communication) is mostly
shipped — dashboard, blog, and docs are live. The authoritative project state lives in
[`docs/STATE.md`](docs/STATE.md), with the post-ship consistency audit in
[`docs/audit-2026-06-10.md`](docs/audit-2026-06-10.md).

Two scope cuts, made as discipline rather than drift. The **90-second demo video** was
**descoped** on 2026-06-07 under the PRD's gate-failure protocol (§9.2) — the dashboard,
blog, and docs already carry the communication load. And **G3** (the Cohen's κ inter-rater
test that validates the failure-mode classifier) is **deferred to v1.1**: the eval harness
never recorded rollout video, so there is nothing for a human rater to blind-label.
Manufacturing labels would defeat the test. The unblock is scoped — a `--record-video`
flag (MuJoCo offscreen render) re-runs the ±5 cm cells, after which the existing scorer in
`scripts/relabel_score.py` runs as designed.

## v1.1 roadmap

- **`--record-video` flag** — unblocks the G3 κ inter-rater test (top priority).
- **Co-trainable α** — lower-bound any residual ablation to "no harm" via an α → 0 collapse.
- **Distillation-init residual** — start PPO from "do nothing" (zero output-layer bias, shrunk weights).
- **ACT-encoder residual input** — feed perceptual features instead of the v1.0 zero-width placeholder.
- **Smaller perturbation cells** (+1, +3 cm) — more headroom for a small additive correction.

## Citation · License · Contact

If you reference this work: *Rubeno Dechua, RoboEval — failure modes & residual RL for ACT, 2026.* <https://github.com/RDechua/roboeval>

Released under the **MIT License** (declared in [`pyproject.toml`](pyproject.toml)).

Contact: rubenodechua123@gmail.com · [github.com/RDechua](https://github.com/RDechua)
