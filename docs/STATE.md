# RoboEval — Current State (2026-05-18, Week 6 Day 1)

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
- **32 source files** in `roboeval/` + `scripts/`, **199 tests passing**.
- CI: ruff + ruff-format + mypy + pytest on push/PR, CPU-only torch wheel.

## Module map

```
roboeval/
├── cli.py                     # smoke, evaluate, calibrate, residual {train,evaluate}.
│                               # evaluate auto-classifies + writes auto_labels JSON
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
    ├── composite.py           # ResidualCompositePolicy (Policy adapter for eval)
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
- **G4 Residual RL** ⏳ Scaffold complete + first training attempt
  diagnosed and aborted. Found two bugs and an explored-vs-exploited
  knob mismatch:
  1. **gym-aloha's nested Dict obs broke SB3** — fixed by exposing a
     flat Box obs (agent_pos + cube_state + base_action + features).
  2. **Double `base.select_action` per env step** — would have
     advanced ACT's chunk pointer 2x per env step. Fixed by caching
     the next base action.
  3. **PPO destroyed the base** at α_init=0.1 + log_std_init=0.0
     (default). With std=1.0 the per-step perturbation magnitude was
     ~0.2 — large enough to push ACT off its narrow successful
     trajectory. ep_rew_mean stayed at 0.0 across 143k steps (0/358
     episodes vs the bare-base 30.7% baseline). Fixed by safer
     defaults in the configs:
     - alpha_init: 0.1 → 0.05
     - log_std_init: 0.0 → -2.0 (std≈0.14)
     train.py now plumbs log_std_init through to PPO's policy_kwargs.
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

Week 6 Day 2 — retry residual training with safer defaults:

1. **First: a 1-minute wrapper sanity check.** Confirms the wrapper
   composition isn't itself the bug (vs PPO learning dynamics):
   ```
   roboeval residual train --config configs/residual/residual_ppo_y+5cm_sparse.yaml
   ```
   Watch the first iteration's ep_rew_mean. With alpha_init=0.05 +
   log_std_init=-2.0 and a 30.7% bare-base TSR, you SHOULD see ~0.3
   ep_rew_mean from iteration 1 (PPO hasn't trained yet; the residual
   contributes <0.01 per dim, so the composed action ≈ base action).
   If you see 0.0, the wrapper is the problem. If you see ~0.3,
   the safer defaults worked and you can let it train.
2. **Run Condition B (sparse) to completion** if step 1 is healthy.
   ~3.5h on M1 at 500k steps. Then 2 more seeds + Condition C.
3. **Evaluate each trained residual** via
   ```
   roboeval residual evaluate --config <same> \
     --residual-path outputs/residual/y+5cm_sparse/ppo_residual.zip
   ```
   Produces auto_labels JSON + W&B summary; PRD §8.3 ablation table
   is a direct A/B/C comparison.
4. **Manual κ relabel** when 2026-05-24 unlock hits — samples are
   already exported (run IDs `alr0r0p2` and `18xb5ob0`).
5. **v1.1 design item**: make alpha co-trainable via custom SB3
   policy. Out of scope; sweep `alpha_init` across runs.
6. **v1.1 design item**: ACT-encoder feature-extractor hook for the
   residual MLP input. Currently residual conditions on privileged
   sim state (agent_pos + cube_state); ACT encoder features would
   make the residual sim-to-real portable.
7. **Optional**: visual / dynamic perturbation axes. Deferred.

## Handoff conventions (preserve across compaction)

- **Patch delivery path**: ALWAYS `/Users/rubenodehcua/Downloads/` —
  NOT `~/Downloads/`. The user has a Mac with this absolute path.
- **Patch filename length**: ≤ 50 chars total (including `.patch`)
  because Safari truncates ~57 chars. I generate a short-named copy
  via `cp /tmp/patches/0001-foo-bar-...patch /tmp/patches/0001short.patch`.
- **Commit author**: `Rubeno Dechua <rubenodechua123@gmail.com>`.
  NO `Co-Authored-By: Claude` trailers anywhere. CLAUDE.md and
  `.claude/` are gitignored.
- **Apply pattern**: `git am /Users/rubenodehcua/Downloads/<file> && git push origin main`.
- **All gates must pass before commit**: ruff check, ruff format,
  mypy --strict, pytest.
- **CI runs minimal deps** (no matplotlib, no SB3). Use
  `pytest.importorskip(...)` to gate tests on optional deps.
- **Headline data** (Phase 3 complete): 7 spatial cells (-5→+5 cm)
  + 3 temporal cells (1/3/5 step delay). Same Recovery-dominant
  failure mode both axes; spatial brittle (67 pp drop at -5cm),
  temporal robust (11 pp at 5 steps). Phase 4 target: +5cm.
- **Open data question**: cross-session `mean_tsr_custom` drift
  (0.680 → 0.727) on nominal cell with same seeds. Mock-env
  determinism test green; real-env audit deferred.
