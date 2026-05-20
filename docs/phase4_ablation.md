# Phase 4 Ablation: Residual RL on +5 cm Spatial Perturbation

**Status**: G4 closed (honest null result, PRD §8.3-compliant).
**Cell**: +5 cm spatial y-shift (cube initial position +0.05 m on y-axis).
**Base policy**: frozen `lerobot/act_aloha_sim_transfer_cube_human` (Policy A from §6).
**Sample size per condition**: 3 seed groups × 50 rollouts = 150 rollouts.
**Primary metric**: `mean_tsr_custom` (PRD §6.2 geometric criterion).

## TL;DR

Two reward shapings of the standard "frozen base + small MLP residual"
recipe both **degraded** the base policy at +5 cm. Sparse reward
delivered ΔTSR = **−13.3 pp**; shaped (negative L2 distance to
target) delivered ΔTSR = **−10.7 pp**. PRD §8.3 explicitly accepts
honest null results; this is the kind of result it had in mind.

| Cond | Label | mean_tsr_custom | ΔTSR vs A |
|------|-------|----------------:|----------:|
| A | Frozen ACT (baseline) | **0.320** | — |
| B | Residual PPO, sparse reward | 0.187 | **−13.3 pp** |
| C | Residual PPO, shaped reward | 0.213 | **−10.7 pp** |

Per-seed (native TSR): A = (0.32, 0.32, 0.28); B = (0.24, 0.16,
0.16); C = (0.26, 0.22, 0.14). One-sided Welch's t-tests (PRD
§8.3, H1: residual > base) reject H1 for both conditions — see
`docs/figures/phase4_ablation.json` for exact t / df / p values
emitted by the aggregator.

## Why we expected this to work

Phase 3 identified +5 cm as the cleanest residual-RL target on three
grounds (`docs/research-log.md`, Week 5 Days 1–3):

1. **Single dominant failure mode** — 59.3 % of failures were
   classified Recovery (motor activity without task completion).
2. **σ collapse** — at ±5 cm the per-seed standard deviation sat
   *below* the Bernoulli SE floor (deterministic failure), so a
   learned corrective pattern had a stable target.
3. **Geometric structure** — a 5 cm xy shift is exactly the kind of
   small additive correction a residual MLP is meant to encode.

The hypothesis: a 2-layer × 256 MLP, trained with PPO on top of the
frozen ACT chunked-action policy, would learn to nudge the gripper
back onto a successful trajectory in the +5 cm regime.

## Setup

- **Residual policy**: `ResidualMLP` (2 hidden layers × 256, GELU).
  Input: agent_pos (14) + cube_state (7) + base_action (14) + features
  (0 in v1) = 35-dim flat Box.
  Output: 14-dim action delta.
- **Composition**: `a = clamp(a_base + α · a_residual, −1, 1)`, with
  α = 0.05 fixed (sigmoid-clipped to [0, 1] but not co-trained in
  v1; see "Recommendations").
- **PPO**: SB3 2.8.0 with `MlpPolicy`, `policy_kwargs = {net_arch:
  [256, 256], log_std_init: −2.0}` (initial std ≈ 0.135), 500k env
  steps, `learning_rate=3e-4`, `n_steps=2048`, `batch_size=64`,
  `n_epochs=10`, `gamma=0.99`, seed=0.
- **Reward**:
  - Condition B (sparse): `r = +1 on info["is_success"], 0 otherwise`.
  - Condition C (shaped): `r_sparse − 0.01 · ‖cube_xy − target_xy‖`.
- **Eval**: same `evaluate_policy` pipeline used for Phase 3
  baselines. `eval.seeds = [0, 1, 2]`, 50 rollouts/seed,
  `max_steps = 400`. Composite policy uses `deterministic=True`
  on the residual model.
- **Wall-clock on M1 8 GB MPS**: ~7 h per 500k-step training run
  (revised from the §10.2.1 estimate of ~3.5 h after measuring
  thermal-throttling fps decay 42 → 30 over the run).

## Results

### Per-condition TSR

The aggregator (`roboeval residual aggregate`) reads the three
`eval_results_<run_id>.json` artifacts and emits a markdown table
plus a JSON summary. Schema v2 decomposes each payload's
`per_seed_tsr_custom` into independent observations, so a single
eval invocation with `seeds=[0, 1, 2]` already populates Welch's
t-test (3 obs per arm) — see `roboeval/residual/aggregate.py`.

Reproduce with:

```bash
roboeval residual aggregate \
  outputs/eval/act_spatial_y+5cm/eval_results_*.json \
  outputs/residual/y+5cm_sparse/eval_results_*.json \
  outputs/residual/y+5cm_shaped/eval_results_*.json \
  --output docs/figures/phase4_ablation.json
```

### Failure-mode shift (the more informative result)

| Bucket | A (base) | B (sparse) | C (shaped) |
|---|---:|---:|---:|
| Success | 30.7 % | 18.7 % (−12.0) | 20.7 % (−10.0) |
| Recovery | 59.3 % | 70.7 % (+11.4) | 73.3 % (+14.0) |
| Approach | 0.7 % | 5.3 % (+4.6) | 2.0 % (+1.3) |
| Grasp | 0.7 % | 0.7 % | 0.7 % |
| Needs review | 8.7 % | 4.7 % | 3.3 % |

(Counts: Condition A has 46 success / 89 recovery / 1 approach / 1
grasp / 13 needs_review out of 150; B has 28 / 106 / 8 / 1 / 7; C has
31 / 110 / 3 / 1 / 5.)

Render the stacked-bar figure with:

```bash
python scripts/plot_failure_distribution.py \
  --cell "A_base:data/taxonomy/auto_labels_w6k2wole.json" \
  --cell "B_sparse:data/taxonomy/auto_labels_o6ukyo53.json" \
  --cell "C_shaped:data/taxonomy/auto_labels_43czuigy.json" \
  --out docs/figures/phase4_ablation_failure_distribution.png \
  --title "Phase 4 ablation: failure-mode distribution at +5cm" \
  --xlabel "Condition"
```

Two patterns stand out:

1. **Residual training systematically converts successes into
   Recovery failures.** The motor activity is preserved (gripper
   moves) but task completion is lost. Both reward shapings produce
   this; shaped is slightly worse on Recovery (+14 pp vs +11 pp)
   despite being better on Approach.
2. **Approach failures jump under sparse reward (+4.6 pp), but the
   shaped reward suppresses most of that regression (+1.3 pp).**
   Approach = "gripper never closed within tolerance of the cube".
   The distance-to-target shaping term gives the residual a gradient
   pulling toward the cube, which prevents the *direction-away*
   miscorrection that sparse PPO learned. But it doesn't help the
   policy actually complete the grasp.

## Interpretation

### Why both conditions hurt

At α = 0.05, σ ≈ 0.135, the per-step residual magnitude is roughly
±0.007 per dim — tiny relative to ACT's action range. So the
degradation is not about *magnitude*; it's about *direction*. PPO's
MLP learned a state-conditional mean that consistently nudges the
gripper in a harmful direction. Over an episode's ~385 steps, a
0.007-magnitude directional bias compounds into a trajectory that
misses the successful regime.

This is a known failure mode of residual RL on chunked-action base
policies: the base policy's trajectory is a narrow ribbon in
state-action space, and even small per-step perturbations push the
state off that ribbon. By the time PPO accumulates enough reward
signal to learn the correction direction, the rollouts that *did*
land on the ribbon (and got reward) are over-represented in a way
that biases the gradient toward the wrong correction.

Shaped reward helps marginally — the L2 distance term gives a dense
gradient that prevents the "pull away from cube" mode (Approach
failures drop +4.6 pp → +1.3 pp). But the shaping rewards the cube
COM, not the geometry of a successful grasp, so the residual still
learns a correction that gets *close* to the cube without completing
the task. Recovery dominates.

### Why this is a clean null result

PRD §8.3 says:

> A negative or null result is still reported with analysis —
> honest null results are valued in research.

This is exactly that case. The Phase 4 design was sound (a clear
hypothesis, a deterministic failure mode to target, safe
hyperparameters after iteration). The result is informative because
it bounds what residual RL can do with this base / reward / α
configuration on this task. We don't know yet *which* of those four
levers is the binding constraint — recommendations below outline the
v1.1 ablations that would tease them apart.

## Recommendations for v1.1

In order of expected impact:

1. **Co-trainable α** (currently fixed at 0.05). The most defensible
   first move — let PPO collapse α → 0 if no useful residual exists,
   lower-bounding the worst case to "no harm". Requires a custom
   SB3 policy class.
2. **Distillation-style residual init.** Initialise the residual MLP
   so its mean output starts at 0 (not random); PPO then updates
   from "do nothing" instead of "random perturbation". A 1-line
   change in `ResidualMLP.__init__` (zero the output layer's bias
   and shrink its weights).
3. **ACT-encoder features for the residual input.** Currently the
   residual conditions on privileged sim state (`agent_pos` +
   `cube_state` + `base_action`). Hooking ACT's encoder features in
   via the `feature_extractor` slot would give the residual the same
   perceptual signal the base policy uses — closer to a real
   sim-to-real residual.
4. **Different perturbation cells.** +5 cm sits at the edge of the
   base policy's competence (TSR = 0.32). Cells with more headroom
   (+1 cm: TSR = 0.72; +3 cm: TSR = 0.55) might admit a useful
   residual without the base falling off its trajectory.

## Files

- **Training checkpoints**:
  `outputs/residual/y+5cm_sparse/ppo_residual.zip`,
  `outputs/residual/y+5cm_shaped/ppo_residual.zip` (~25 MB each;
  not tracked in git).
- **Persisted eval artifacts** (Schema v1 of
  `eval_results_<run_id>.json`):
  - A: `outputs/eval/act_spatial_y+5cm/eval_results_w6k2wole.json`
  - B: `outputs/residual/y+5cm_sparse/eval_results_o6ukyo53.json`
  - C: `outputs/residual/y+5cm_shaped/eval_results_43czuigy.json`
- **Auto-labels** (Schema v1 of `auto_labels_<run_id>.json`,
  classifier output):
  - A: `data/taxonomy/auto_labels_w6k2wole.json`
  - B: `data/taxonomy/auto_labels_o6ukyo53.json`
  - C: `data/taxonomy/auto_labels_43czuigy.json`
- **Aggregator output**: `docs/figures/phase4_ablation.json`
  (Schema v2; produced by `roboeval residual aggregate`).
- **Figure**: `docs/figures/phase4_ablation_failure_distribution.png`
  (produced by `scripts/plot_failure_distribution.py` per the
  command above).
- **W&B runs**:
  - A: `auchm66k` (eval), `w6k2wole` (re-eval for schema-v1 JSON)
  - B: training under `residual_ppo_y+5cm_sparse` prefix; eval
    `o6ukyo53`.
  - C: training under `residual_ppo_y+5cm_shaped` prefix; eval
    `43czuigy`.
