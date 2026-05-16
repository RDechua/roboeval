# RoboEval — Current State (2026-05-15, Week 4 Day 1)

A tight session-handoff anchor. `docs/PRD.md` is "what we're building",
`docs/research-log.md` is "what happened week-by-week", this file is
"where things stand right now". Update on every major commit; deletions
welcome (it's a snapshot, not a journal).

## Phase

Phase 3 (robustness study) in progress. Gates G1 (foundation) and G2
(baseline) closed. Currently between Week 4 (spatial perturbation cells)
and Week 5 (trajectory-data extension → failure-mode classifier).

## Headline numbers

- ACT baseline (`act_nominal.yaml`, 3 seeds × 50 rollouts, M1 MPS):
  `mean_tsr = 0.80 ± 0.057` (per_seed 88, 76, 76). Model card: 0.83
  (415/500 sequential seeds). Δ within sampling noise — G2 validated.
- `mean_tsr_custom = 0.68 ± 0.075` (PRD §6.2 geometric criterion;
  ~10 pp gap is the 90th-percentile calibration tail by design).
- Calibration: `target_xy = (-0.01835, 0.50576)`, `xy_tolerance_m = 0.02185`.
  Frozen at `data/calibration/transfer_cube_target_xy.json`.
  `dwell_steps = 1` (gym-aloha terminates on `reward==4` same-step).

## Spatial degradation curve (Week 4)

| cell | mean_tsr | σ |
|---|---|---|
| nominal | 0.80 | 0.057 |
| y+1cm   | 0.72 | 0.102 |
| y+3cm   | 0.55 | 0.041 |
| y+5cm   | 0.31 | 0.019 |

Key finding: σ peaks at +1cm then collapses **below the Bernoulli noise floor**.
Competence-collapse signature — worth a paragraph in the §6.4 writeup.

## Stack / repo state

- Python 3.11, `mypy --strict`, ruff, `lerobot==0.4.4`, MuJoCo+gym-aloha,
  Stable-Baselines3, Hydra (OmegaConf), W&B, Plotly+Dash.
- **22 source files** in `roboeval/`, **82 tests passing**.
- CI: ruff + ruff-format + mypy + pytest on push/PR, CPU-only torch wheel.

## Module map

```
roboeval/
├── cli.py                     # 3 subcommands: smoke, evaluate, calibrate
├── envs/aloha.py              # LeRobot env factory + get_cube_state
├── envs/success.py            # geometric detector; no defaults, all fields required
├── envs/perturb.py            # SpatialShiftWrapper + factory (visual/dynamic/temporal stubbed)
├── evaluation/types.py        # RolloutResult, EvalResult (frozen dataclasses)
├── evaluation/rollout.py      # run_rollout with NaN/Inf/bound guards
├── evaluation/loop.py         # multi-seed evaluate_policy
├── evaluation/calibration.py  # calibrate_target_xy + ${calibration:...} resolver
├── evaluation/config.py       # load_eval_config with extends: support
├── evaluation/logger.py       # W&B context manager + artifact upload
├── policies/base.py           # Policy Protocol (declares policy_id, device)
├── policies/act_loader.py     # ACTPolicyAdapter, lazy lerobot imports
├── policies/factory.py        # load_policy(kind, ...); diffusion = v1.1
├── taxonomy/types.py          # FailureMode enum, RolloutLabel
├── taxonomy/classifier.py     # Timeout + NEEDS_REVIEW; trajectory rules deferred
├── taxonomy/agreement.py      # Cohen's κ for §7.3 blinded relabel
└── residual/                  # empty — Weeks 6–7

configs/
├── baseline/act_nominal{,_fast,_mps_check}.yaml
└── perturbation/spatial/act_spatial_y+{1,3,5}cm.yaml

data/calibration/transfer_cube_target_xy.json     # frozen calibration artifact
```

## Quality gates (PRD §9.2)

- **G1 Foundation** ✓ CI green, MPS verified, smoke runs
- **G2 Baseline** ✓ 80% TSR within ±5 pp of model card 83%, σ=5.7% < 7%,
  policy-agnostic via `policy.kind`
- **G3 Robustness & Taxonomy** ⏳ spatial 3 of ~6 cells; visual/dynamic/temporal
  not started; trajectory data → classifier branches pending
- **G4 Residual RL** — not started (Weeks 6–7)
- **G5 Communication** — not started (Week 8)
- **G6 Launch** — not started (Weeks 9–10)

## Active conventions / decisions

- Commit author: `Rubeno Dechua <rubenodechua123@gmail.com>`. **No
  `Co-Authored-By` trailers; no CLAUDE.md or .claude/ tracked** (gitignored).
- All edits in remote sessions arrive as `git format-patch` files via the
  chat UI. **Safari strips dashes from filenames on download** — use
  `ls ~/Downloads | grep -i <keyword>` to find exact names.
- Every commit must pass `ruff check`, `ruff format --check`, `mypy --strict`,
  `pytest -q`.
- Reproducibility: bit-identical `(success, n_steps, success_custom)` across
  same-seed runs on M1 MPS — confirmed Week 3.

## Open / deferred

- **Trajectory data on `RolloutResult`** (action sign-flip rate, terminal EE
  pose, contact bit, last-50-step cube displacement) — **Week 5 priority**.
  Lights up 4 currently-`NEEDS_REVIEW` classifier branches (Grasp, Approach,
  Oscillation, Recovery). Visual Confusion is an aggregate-over-runs rule,
  handled separately.
- Negative spatial shifts (y−1, −3, −5 cm).
- Visual / dynamic / temporal perturbation wrappers (stubs in place).
- `normalize_inputs.*` warning — diagnosed harmless, optional cleanup via
  upstream `lerobot/processor/migrate_policy_normalization.py`.
- W&B artifact-audit: confirm config artifact publicly downloadable from
  run `tlbkwp5o`.

## Next session intent

Week 5 trajectory-data extension, in this order:

1. Add 4 aggregate fields to `RolloutResult` + compute in `run_rollout`.
2. Light up the 4 classifier branches in `taxonomy/classifier.py`.
3. Re-run the existing 3 spatial cells (~75 min total on M1) →
   failure-mode distribution per cell.
4. Then negative shifts + remaining perturbation axes, classifier already in
   place.
