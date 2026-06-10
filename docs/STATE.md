# RoboEval — Current State (2026-06-10, Week 9, G3 deferred to v1.1)

A tight session-handoff anchor. `docs/PRD.md` is "what we're building",
`docs/research-log.md` is "what happened week-by-week", this file is
"where things stand right now". Update on every major commit; deletions
welcome (it's a snapshot, not a journal).

## Phase

**Phase 4 closed**: residual RL ablation complete at +5 cm cell;
honest null result documented in `docs/phase4_ablation.md`. Gates
G1, G2, G4 closed. **G3 deferred to v1.1** (2026-06-10): the embargo
unlocked 2026-05-24 but the manual relabel can't run — the eval
harness never recorded rollout video to W&B or disk, so no human can
watch and label the sampled rollouts. Unblock = add a `--record-video`
flag to `roboeval evaluate` (mujoco offscreen render +
`imageio.mimsave`), re-run the ±5 cm cells, then the existing
`scripts/relabel_score.py` pipeline runs as designed. See
`docs/kappa-relabel-runbook.md` (now banner-marked "won't run") and
the 2026-06-10 research-log entry.

**Phase 5 (Communication) in progress**: interactive Plotly/Dash
dashboard landed (`roboeval/dashboard/` + `analysis/dashboard/`),
narrative single-page hero curves + per-cell failure breakdown +
Phase 4 ablation. Deploys to HF Spaces via `analysis/dashboard/Dockerfile`.
arXiv-style writeup remains; the 90-second demo video was descoped
2026-06-07 (PRD §3.2 non-goals, gate-failure protocol §9.2).

`data/headline.json` is schema v2 — self-contained (cells + ablation +
Welch's t blocks inline). Runtime reads one tracked file; gitignored
auto_labels / eval_results are touched only at build time by
`scripts/build_headline_json.py`. See
`docs/superpowers/specs/2026-05-21-phase5-dashboard-design.md` for the
schema-v1→v2 amendment.

Cross-phase finding still standing: ACT's failure mode is
policy-architecture-specific (both spatial and temporal axes produce
Recovery), but elasticity differs 4-6× (spatial brittle, temporal
robust). Residual RL on +5 cm was the cleanest target identified by
Phase 3, but the trained residual hurt the base by 13.3 pp (sparse)
/ 10.7 pp (shaped) — see `docs/phase4_ablation.md` for the analysis
and v1.1 recommendations.

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
- **41 source files** in `roboeval/` + `scripts/` (34 + 7), **285 tests passing**.
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
├── evaluation/results_io.py   # schema-v1 eval_results_<run_id>.json writer (NEW)
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
    ├── train.py               # SB3 PPO training loop
    └── aggregate.py           # PRD §8.3 ablation: Welch's t-test +
                                #   bootstrap CI, stdlib-only (NEW)

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
docs/figures/phase4_ablation.json               # PRD §8.3 aggregator output
docs/phase4_ablation.md                         # PRD §8.3 G4 writeup
```

## Quality gates (PRD §9.2)

- **G1 Foundation** ✓ CI green, MPS verified, smoke runs
- **G2 Baseline** ✓ 80% TSR within ±5 pp of model card 83%, σ=5.7% < 7%
- **G3 Robustness & Taxonomy** ⏸ **deferred to v1.1 (2026-06-10).**
  Spatial 7/7 cells + temporal 3/3 cells run with full auto-classifier
  output; all 6 classifier rules wired + auto-labels artifact produced
  per eval run; PRD §7.3 step 4 relabel-sample exporter live; ±5 cm
  samples exported (embargo unlocked 2026-05-24). Blocked at the
  manual-label step: the eval harness never logged or rendered rollout
  video (no `wandb.Video`, no `imageio`, no MP4s in `outputs/`), so no
  human can watch the sampled rollouts to assign blind labels.
  Unblock = add `--record-video` to `roboeval evaluate`, re-run ±5 cm,
  then run `scripts/relabel_score.py` against the existing samples.
- **G4 Residual RL** ✓ Closed 2026-05-20. Honest null result per
  PRD §8.3: ΔTSR = −13.3 pp (sparse), −10.7 pp (shaped) vs frozen
  base at +5 cm. Full writeup in `docs/phase4_ablation.md`. Six
  intermediate findings + fixes:
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
  4. **Ablation aggregator ready (PRD §8.3 deliverable)**.
     `roboeval residual aggregate <eval_results_*.json>` reads N
     persisted run summaries, classifies by condition (A=frozen base,
     B=sparse, C=shaped), and emits per-condition mean±std + 95%
     bootstrap CI + one-sided Welch's t-test per non-baseline
     condition vs A. Stdlib-only math (Student-t CDF via
     incomplete-beta Lentz CF) so the aggregator runs in CI without
     scipy/numpy. Both `roboeval evaluate` and
     `roboeval residual evaluate` now persist
     `eval_results_<run_id>.json` next to their other artifacts so the
     aggregator has something to read. Schema v2: per-seed
     decomposition — each payload's `per_seed_tsr_custom` list
     contributes N independent observations to its condition, so a
     single eval run with `eval.seeds = [0, 1, 2]` already populates
     Welch's t-test (3 obs per arm) instead of needing 3 separate
     `evaluate` invocations.
  5. **Eval-time obs-format bug fixed**. PPO trained on the wrapper's
     flat Box(35,) obs; the original `ResidualCompositePolicy` was
     passing the raw gym-aloha Dict to `model.predict`, crashing
     SB3's `obs_to_tensor` with "The observation provided is a dict
     but the obs space is Box(-inf, inf, (35,), float32)". Fixed by
     extracting `build_flat_obs` to module top in `env_wrapper.py` so
     train + eval share one builder, then adding an `obs_builder`
     hook to `ResidualCompositePolicy` and wiring the eval CLI to
     construct it bound to the shared env + the same feature
     extractor used at training. Regression test in
     `tests/residual/test_composite.py`.
  6. **Trained, evaluated, aggregated.** Condition A (frozen ACT
     at +5 cm): mean_tsr_custom = 0.320. Condition B (sparse,
     500 k steps, ~7 h M1): 0.187 → ΔTSR −13.3 pp. Condition C
     (shaped, 500 k steps, ~7 h M1): 0.213 → ΔTSR −10.7 pp.
     Per-seed and failure-mode breakdowns in
     `docs/phase4_ablation.md`. Recovery share grows from 59 %
     (A) to 71 % (B) / 73 % (C); residual training systematically
     converts successes into Recovery failures. Honest null per
     PRD §8.3; v1.1 recommendations (co-trainable α, distillation
     init, ACT-encoder features) listed in the writeup.
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

Phase 5 mostly shipped (2026-05-22). What's left:

1. **HF Spaces dashboard landed** ✓ — live at
   https://rubenodechua-roboeval.hf.space/ (canonical lowercase HF
   username `rubenodechua`; canonical GitHub username `RDechua`).
   Auto-redeploys on push to `main` of the Space repo. Dockerfile uses
   `pip install --no-deps -e .` to fit the free-tier disk budget. Mirror
   recipe in memory: `feedback_hf_spaces.md`.
2. **Blog post landed** ✓ — `docs/blog/2026-05-21-honest-null-residual.md`,
   ~2090 words, honest-null framing, three figures. Linked from README.
   arXiv-style PDF cross-post deferred.
3. **MkDocs site landed** ✓ — live at https://rdechua.github.io/roboeval/.
   Lean 4-tab nav, 7 mkdocstrings submodule pages, auto-deploy via
   `.github/workflows/docs.yml`. Repo is public so Pages serves for free.
4. **90-second demo video** — **descoped 2026-06-07** per PRD §9.2
   gate-failure protocol; recorded in PRD §3.2 non-goals. Script doc
   and capture assets removed from the repo.
5. **κ relabel for G3 — deferred to v1.1 (2026-06-10).** Embargo
   unlocked 2026-05-24 as planned, but the runbook can't execute:
   the eval harness never recorded rollout video (no `wandb.Video`
   calls anywhere in `roboeval/`, no MP4s on disk). The relabel
   sample tells the labeller "watch rollout (seed_group, rollout_idx)
   and pick a category" — there's nothing to watch. The auto-classifier
   worked fine without video (it operates on trajectory metrics), but
   the **manual** step the κ test requires can't proceed.
   - **Unblock plan (v1.1, ~3-4 h):** add a `--record-video` flag to
     `roboeval evaluate` using mujoco's offscreen renderer
     (`render_mode="rgb_array"` + `imageio.mimsave`), re-run the ±5 cm
     cells (~20 min each), then watch+label+score against the existing
     samples at `data/taxonomy/relabel_sample_{18xb5ob0,alr0r0p2}.json`.
   - **Scorer + runbook stay in the repo** as the future-proof landing
     point. `scripts/relabel_score.py` (4 unit tests, all green) is
     correct; `docs/kappa-relabel-runbook.md` now carries a "won't run
     yet" banner pointing at this v1.1 plan.

Dashboard implementation notes:

- Pure logic under `roboeval/dashboard/` (mypy --strict,
  unit-tested in `tests/dashboard/`); Dash skeleton in
  `analysis/dashboard/app.py`.
- Single committed `data/headline.json` is the runtime source of
  truth for 10 Phase 3 cells; built by
  `scripts/build_headline_json.py` from gitignored auto_labels
  + STATE.md headline tables.
- Two CLI surfaces: `roboeval dashboard build` (CI data-gate) and
  `roboeval dashboard run` (local dev server on :8050).
- HF Spaces uses the Docker SDK; container CMD runs gunicorn on
  port 7860. First visit on a cold container is ~30 s; warm
  loads are <3 s per PRD §9.1.

v1.1 design backlog (deferred from Phase 4):

- **Co-trainable α** via custom SB3 policy class. Top priority —
  lower-bounds residual to "no harm".
- **Distillation-init residual**: zero output-layer bias and shrink
  weights so PPO starts from "do nothing". 1-line change in
  `ResidualMLP.__init__`.
- **ACT-encoder features** via the existing `feature_extractor`
  slot; would make the residual sim-to-real portable.
- **Smaller perturbation cells** (+1, +3 cm) where the base has
  more headroom for an additive correction.
- **Visual / dynamic perturbation axes** — wrappers stubbed.

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
