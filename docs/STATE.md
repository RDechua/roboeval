# RoboEval — Current State (2026-05-17, Week 5 Day 2)

A tight session-handoff anchor. `docs/PRD.md` is "what we're building",
`docs/research-log.md` is "what happened week-by-week", this file is
"where things stand right now". Update on every major commit; deletions
welcome (it's a snapshot, not a journal).

## Phase

Phase 3 (robustness study) in progress. Gates G1 (foundation) and G2
(baseline) closed. Week 5 trajectory-data extension + classifier wire-up
complete; **full spatial axis run (-5 → +5 cm, 7 cells)** with failure-
mode distribution per cell. Temporal axis wrapper + 3 configs scaffolded
but not yet run on M1.

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

## Spatial degradation curve + failure-mode distribution (Weeks 4-5, 7 cells)

| cell | mean_tsr | σ | Success | Recovery | Approach | Timeout | Needs review |
|---|---|---|---|---|---|---|---|
| **-5cm** | 0.127 | 0.009 | 12.7% | **80.7%** | 4.7% | 0 | 2.0% |
| -3cm     | 0.553 | 0.025 | 55.3% | **42.0%** | 0    | 0 | 2.7% |
| -1cm     | 0.827 | 0.034 | 82.7% | 17.3%     | 0    | 0 | 0    |
| nominal  | 0.800 | 0.057 | 80.0% | 0%        | 0    | 18.7% | 1.3% |
| +1cm     | 0.720 | 0.102 | 72.0% | 24.7%     | 0    | 0 | 3.3% |
| +3cm     | 0.553 | 0.041 | 55.3% | 37.3%     | 0.7% | 0 | 6.7% |
| **+5cm** | 0.307 | 0.019 | 30.7% | **59.3%** | 0.7% | 0 | 8.7% |

Figure at `docs/figures/spatial_failure_distribution.png` (7-cell stack).
Headline findings (research-log Week 5 Day 2 for full analysis):

1. **Asymmetric curve** — at ±1 cm, −1cm is +2.7 pp *better* than nominal
   while +1cm is −8.0 pp *worse*. At ±3 cm both match. At ±5 cm the
   asymmetry has flipped: −5cm (12.7%) is dramatically worse than +5cm
   (30.7%).
2. **Recovery dominates** failures across both directions.
3. **Approach failures emerge at −5cm** (4.7% vs 0.7% at +5cm) — the
   first per-cell qualitative shift in failure morphology.
4. **σ collapse is more aggressive on the negative side** — sub-Bernoulli
   from −1cm onward; positive side is super-Bernoulli at +1cm before
   collapsing. Negative-y is deterministic; positive-y is variable at
   small magnitude then collapses.

Phase 4 base-policy target: **+5cm remains the cleanest** (single failure
mode, 59% Recovery, deterministic). −5cm has more headroom but multi-modal
failure (Recovery + Approach), harder residual-RL signal.

## Stack / repo state

- Python 3.11, `mypy --strict`, ruff, `lerobot==0.4.4`, MuJoCo+gym-aloha,
  Stable-Baselines3, Hydra (OmegaConf), W&B, matplotlib, Plotly+Dash.
- **27 source files** in `roboeval/` + `scripts/`, **153 tests passing**.
- CI: ruff + ruff-format + mypy + pytest on push/PR, CPU-only torch wheel.

## Module map

```
roboeval/
├── cli.py                     # smoke, evaluate, calibrate. evaluate now
│                               # auto-classifies + writes auto_labels JSON
├── envs/aloha.py              # env factory + cube_state + gripper_xy + contact
├── envs/success.py            # geometric detector; no defaults, all fields required
├── envs/perturb.py            # Spatial + Temporal wrappers; visual/dynamic stubbed
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
├── relabel_from_wandb.py         # post-hoc relabel a completed W&B run
├── plot_failure_distribution.py  # stacked-bar PNG (panel A of §6.4 figure)
├── plot_degradation_curve.py     # TSR-vs-x + Bernoulli SE floor (panel B)
└── export_relabel_sample.py      # PRD §7.3 step 4 redacted sample exporter

configs/
├── baseline/act_nominal{,_fast,_mps_check}.yaml
├── perturbation/spatial/act_spatial_y{+,-}{1,3,5}cm.yaml  # 6 cells, -5 to +5 cm
└── perturbation/temporal/act_temporal_delay_{1,3,5}steps.yaml  # 3 cells

data/
├── calibration/transfer_cube_target_xy.json    # frozen calibration artifact
└── taxonomy/auto_labels_<run_id>.json          # frozen evidence trail (gitignored)

docs/figures/spatial_failure_distribution.png   # §6.4 panel A (stacked-bar)
docs/figures/spatial_degradation_curve.png      # §6.4 panel B (TSR + SE floor)
```

## Quality gates (PRD §9.2)

- **G1 Foundation** ✓ CI green, MPS verified, smoke runs
- **G2 Baseline** ✓ 80% TSR within ±5 pp of model card 83%, σ=5.7% < 7%
- **G3 Robustness & Taxonomy** ⏳ spatial 7/7 cells run (−5 → +5 cm
  complete); temporal wrapper + 3 cells scaffolded (not yet run);
  visual/dynamic not started; all 6 classifier rules wired + auto-labels
  artifact produced per eval run
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

Week 5 Day 3 — temporal axis runs + multi-axis writeup prep:

1. Run the 3 temporal cells on M1 (~75 min):
   `roboeval evaluate --config configs/perturbation/temporal/act_temporal_delay_{1,3,5}steps.yaml`.
   Auto-classify writes labels per cell; no extra code needed.
2. Render a temporal-axis figure (re-use `scripts/plot_failure_distribution.py`
   with the 3 temporal cells + nominal). Likely a different stack
   shape than spatial — action delay may produce more Oscillation /
   Approach / mixed failures than Recovery, since the delayed-action
   stream effectively asks the policy to operate on stale observations.
3. Draft a 2-axis §6.4 paragraph comparing spatial (Recovery-dominant,
   asymmetric) vs temporal (??). The contrast is the deliverable that
   gives §6.4 a real cross-axis perspective rather than just
   "spatial works".
4. **Open question still**: cross-session `mean_tsr_custom` drift
   (nominal: 0.680 → 0.727). Determinism regression on mock env passes;
   real-env audit needs a same-session double-run on M1. Low priority
   relative to the multi-axis writeup.
