# RoboEval — Current State (2026-05-17, Week 5 Day 3)

A tight session-handoff anchor. `docs/PRD.md` is "what we're building",
`docs/research-log.md` is "what happened week-by-week", this file is
"where things stand right now". Update on every major commit; deletions
welcome (it's a snapshot, not a journal).

## Phase

Phase 3 (robustness study) in progress. Gates G1 (foundation) and G2
(baseline) closed. Week 5 trajectory-data extension + classifier wire-up
complete; **full spatial axis (−5 → +5 cm, 7 cells) and temporal axis
(1/3/5 step delay, 3 cells) both run** with failure-mode distributions.
Cross-axis finding: ACT's failure mode is policy-architecture-specific
(both axes produce Recovery), but elasticity differs 4-6× (spatial brittle,
temporal robust). Phase 4 base-policy target: +5cm.

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

## Temporal degradation (Week 5, 3 cells + nominal)

| delay | TSR | σ | Success | Recovery | Needs review |
|---|---|---|---|---|---|
| nominal | 0.800 | 0.057 | 80.0% | 0 (28 Timeout) | 1.3% |
| 1 step  | 0.753 | 0.050 | 75.3% | 22.0% | 2.7% |
| 3 steps | 0.767 | 0.068 | 76.7% | 21.3% | 2.0% |
| 5 steps | 0.687 | 0.066 | 68.7% | 30.0% | 1.3% |

Figures: `docs/figures/temporal_{failure_distribution,degradation_curve}.png`.
Same Recovery-dominant failure mode as spatial; **TSR loses only 11 pp
at 5-step delay vs 49 pp at +5cm spatial**. ACT's 100-step action chunking
+ temporal ensembling absorbs most of the latency. σ stays super-Bernoulli
throughout — no competence collapse on this axis.

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
- **31 source files** in `roboeval/` + `scripts/`, **183 tests passing**.
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
└── residual/                  # Phase 4: residual RL primitives
    ├── policy.py              # ResidualMLP (2x256 GELU) + ResidualCompositor
    ├── reward.py              # sparse + shaped + combined reward functions
    ├── env_wrapper.py         # gym wrapper composing base + residual
    └── train.py               # SB3 PPO training loop

scripts/
├── relabel_from_wandb.py         # post-hoc relabel a completed W&B run
├── plot_failure_distribution.py  # stacked-bar PNG (panel A of §6.4 figure)
├── plot_degradation_curve.py     # TSR-vs-x + Bernoulli SE floor (panel B)
└── export_relabel_sample.py      # PRD §7.3 step 4 redacted sample exporter

configs/
├── baseline/act_nominal{,_fast,_mps_check}.yaml
├── perturbation/spatial/act_spatial_y{+,-}{1,3,5}cm.yaml  # 6 cells, -5 to +5 cm
├── perturbation/temporal/act_temporal_delay_{1,3,5}steps.yaml  # 3 cells
└── residual/residual_ppo_y+5cm_{sparse,shaped}.yaml  # Phase 4 Conditions B,C

data/
├── calibration/transfer_cube_target_xy.json    # frozen calibration artifact
└── taxonomy/auto_labels_<run_id>.json          # frozen evidence trail (gitignored)

docs/figures/spatial_failure_distribution.png   # §6.4 panel A (stacked-bar)
docs/figures/spatial_degradation_curve.png      # §6.4 panel B (TSR + SE floor)
docs/figures/temporal_failure_distribution.png  # §6.4 temporal-axis panel A
docs/figures/temporal_degradation_curve.png     # §6.4 temporal-axis panel B
```

## Quality gates (PRD §9.2)

- **G1 Foundation** ✓ CI green, MPS verified, smoke runs
- **G2 Baseline** ✓ 80% TSR within ±5 pp of model card 83%, σ=5.7% < 7%
- **G3 Robustness & Taxonomy** ⏳ spatial 7/7 cells + temporal 3/3 cells
  run with full classifier output; visual/dynamic not started; all 6
  classifier rules wired + auto-labels artifact produced per eval run.
  PRD §7.3 step 4 relabel-sample exporter live; +5cm and -5cm samples
  exported, unlock 2026-05-24.
- **G4 Residual RL** ⏳ Phase 4 scaffold complete (`roboeval/residual/`:
  ResidualMLP per PRD §8.2 spec, ResidualCompositor with learnable alpha,
  env wrapper, SB3 PPO training entry point). Configs for Conditions B
  (sparse) and C (shaped) at +5cm target. Training run not started yet.
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

Week 6 — kick off Phase 4 residual RL training:

1. **Add `roboeval residual train --config` CLI subcommand** that
   wraps `roboeval.residual.train.train_residual`. Loads the residual
   config block (alpha_init, total_timesteps, learning_rate, etc.),
   constructs the base policy via `load_policy`, builds the env
   factory with the spatial perturbation, picks the reward function
   from the config's `residual.reward.kind`, runs PPO, saves the
   trained model to `outputs/residual/<run_name>/`. Goal: one command
   per ablation condition.
2. **ACT-encoder feature-extractor hook.** Currently `zero_feature_
   extractor` returns a 0-dim vector; residual conditions only on the
   base action. Wire up a hook into the ACT policy's encoder forward
   so the residual sees the actual perceptual features. (Lerobot
   internals; ~half-day of careful integration.)
3. **Run Condition B (sparse) first** for the smallest seed (~1h on
   M1 at 500k steps with action-chunked inference). Sanity-check
   training curves in W&B; if PPO is learning anything, run the
   remaining 2 seeds + the 3 Condition C seeds.
4. **Manual κ relabel** when 2026-05-24 unlock hits — same plan as
   before; samples are already exported (`alr0r0p2` and `18xb5ob0`).
5. **Optional**: visual / dynamic perturbation axes. Deferred per
   Week 5 Day 3 conclusion — the spatial+temporal pair already
   supplies the cross-axis comparison; adding axes adds polish.
