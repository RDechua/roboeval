# RoboEval — Current State (2026-05-17, Week 5 Day 1)

A tight session-handoff anchor. `docs/PRD.md` is "what we're building",
`docs/research-log.md` is "what happened week-by-week", this file is
"where things stand right now". Update on every major commit; deletions
welcome (it's a snapshot, not a journal).

## Phase

Phase 3 (robustness study) in progress. Gates G1 (foundation) and G2
(baseline) closed. Week 5 trajectory-data extension + classifier wire-up
complete; failure-mode distribution computed for the 3 spatial cells.
Next: negative spatial shifts, then remaining perturbation axes.

## Headline numbers

- ACT baseline (`act_nominal.yaml`, 3 seeds × 50 rollouts, M1 MPS):
  `mean_tsr = 0.80 ± 0.057` (per_seed 88, 76, 76). Model card: 0.83
  (415/500 sequential seeds). Δ within sampling noise — G2 validated.
- `mean_tsr_custom = 0.727 ± 0.068` (PRD §6.2 geometric criterion;
  drifted from 0.680 across sessions — probably the new physics reads in
  `run_rollout` advancing dm_control state. Within sampling noise but
  worth a reproducibility test next session.)
- Calibration: `target_xy = (-0.01835, 0.50576)`, `xy_tolerance_m = 0.02185`.
  Frozen at `data/calibration/transfer_cube_target_xy.json`.
  `dwell_steps = 1` (gym-aloha terminates on `reward==4` same-step).

## Spatial degradation curve + failure-mode distribution (Weeks 4-5)

| cell | mean_tsr | σ | Success | Recovery | Timeout | Needs review |
|---|---|---|---|---|---|---|
| nominal | 0.80 | 0.057 | 80.0% | 0%    | **18.7%** | 1.3% |
| y+1cm   | 0.72 | 0.102 | 72.0% | **24.7%** | 0%    | 3.3% |
| y+3cm   | 0.55 | 0.041 | 55.3% | **37.3%** | 0%    | 6.7% |
| y+5cm   | 0.31 | 0.019 | 30.7% | **59.3%** | 0%    | 8.7% |

Grasp/Approach/Oscillation are ≤ 1.3% across all cells. Figure at
`docs/figures/spatial_failure_distribution.png`. Headline: **under any
positive y-shift the failure mode flips from Timeout to Recovery**;
policy ends within 5 cm of cube, never engages, stays quiet. ACT has no
learned response to off-nominal cube placement. **+5cm is the strongest
Phase 4 residual-RL target** (59% Recovery, deterministic, geometric).

## Stack / repo state

- Python 3.11, `mypy --strict`, ruff, `lerobot==0.4.4`, MuJoCo+gym-aloha,
  Stable-Baselines3, Hydra (OmegaConf), W&B, matplotlib, Plotly+Dash.
- **24 source files** in `roboeval/` + `scripts/`, **110 tests passing**.
- CI: ruff + ruff-format + mypy + pytest on push/PR, CPU-only torch wheel.

## Module map

```
roboeval/
├── cli.py                     # smoke, evaluate, calibrate. evaluate now
│                               # auto-classifies + writes auto_labels JSON
├── envs/aloha.py              # env factory + cube_state + gripper_xy + contact
├── envs/success.py            # geometric detector; no defaults, all fields required
├── envs/perturb.py            # SpatialShiftWrapper + factory (visual/dyn/temp stubbed)
├── evaluation/types.py        # RolloutResult (20 fields incl. 4 trajectory aggregates)
├── evaluation/rollout.py      # run_rollout w/ trajectory-aggregate helpers; injectable
├── evaluation/loop.py         # multi-seed evaluate_policy
├── evaluation/calibration.py  # calibrate_target_xy + ${calibration:...} resolver
├── evaluation/config.py       # load_eval_config with extends: support
├── evaluation/logger.py       # W&B ctx mgr + log_distribution + run_id; 20-col table
├── policies/base.py           # Policy Protocol (declares policy_id, device)
├── policies/act_loader.py     # ACTPolicyAdapter, lazy lerobot imports
├── policies/factory.py        # load_policy(kind, ...); diffusion = v1.1
├── taxonomy/types.py          # FailureMode enum, RolloutLabel
├── taxonomy/classifier.py     # All 6 PRD §7.2 rules + priority + perturbation_applied
├── taxonomy/agreement.py      # Cohen's κ for §7.3 blinded relabel
├── taxonomy/io.py             # schema-v1 auto_labels_<run_id>.json writer
└── residual/                  # empty — Weeks 6–7

scripts/
├── relabel_from_wandb.py        # post-hoc relabel a completed W&B run
└── plot_failure_distribution.py # stacked-bar PNG generator for §6.4 figure

configs/
├── baseline/act_nominal{,_fast,_mps_check}.yaml
└── perturbation/spatial/act_spatial_y+{1,3,5}cm.yaml

data/
├── calibration/transfer_cube_target_xy.json    # frozen calibration artifact
└── taxonomy/auto_labels_<run_id>.json          # frozen evidence trail (gitignored)

docs/figures/spatial_failure_distribution.png   # §6.4 headline figure
```

## Quality gates (PRD §9.2)

- **G1 Foundation** ✓ CI green, MPS verified, smoke runs
- **G2 Baseline** ✓ 80% TSR within ±5 pp of model card 83%, σ=5.7% < 7%
- **G3 Robustness & Taxonomy** ⏳ spatial 3/~6 cells with full taxonomy
  distribution; visual/dynamic/temporal not started; all 6 classifier
  rules wired + auto-labels artifact produced per eval run
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
- `data/taxonomy/auto_labels_*.json` is gitignored (PRD §7.3 frozen but
  per-run, regeneratable from W&B via `scripts/relabel_from_wandb.py`).

## Open / deferred

- **Reproducibility regression test** — assert bit-identical trajectory
  aggregates across same-seed re-runs on mock env. Addresses the
  `mean_tsr_custom` drift noted above. **Next session priority.**
- **Negative spatial shifts** (y-1, y-3, y-5 cm) — symmetrise the curve,
  test directional sensitivity.
- **Visual / dynamic / temporal perturbation wrappers** (stubs in place).
- **Manual audit of needs_review rollouts** (~5 from +5cm cell) — decide
  whether to tighten Recovery's sign-flip threshold or split off a new
  "motion without engagement" category.
- **`normalize_inputs.*` warning** — diagnosed harmless, optional cleanup via
  upstream `lerobot/processor/migrate_policy_normalization.py`.
- **W&B artifact-audit**: confirm config artifact publicly downloadable.

## Next session intent

Week 5 Day 2 — close reproducibility + scale the perturbation suite:

1. Write `tests/evaluation/test_rollout_aggregates_deterministic.py`:
   assert bit-identical trajectory aggregates across two same-seed runs
   on the mock env. Reproduces the cross-session drift in
   `mean_tsr_custom` or rules out the new physics reads as cause.
2. Add 3 negative spatial config cells (`act_spatial_y-{1,3,5}cm.yaml`)
   + run them on M1 (~75 min). Same auto-classify path, no new code.
3. Re-render `docs/figures/spatial_failure_distribution.png` to span
   −5 → +5 cm. Updates §6.4 in research-log with directional findings.
4. Pick the next perturbation axis: temporal (cheapest — action
   downsampling/upsampling) or visual (needs render-pipeline hooks).
