# RoboEval Perturbation Suite — Config Layout

Implements PRD §6.4. Each config is one (axis, intensity) cell. Cells inherit from `configs/baseline/act_nominal.yaml` via the `extends:` key resolved by `roboeval.evaluation.config.load_eval_config`, so they only need to override the `perturbation:` block and the W&B name/tags.

Run a cell with: `roboeval evaluate --config configs/perturbation/<axis>/<cell>.yaml`. Wall-time per cell on M1 MPS is ~20–25 minutes (3 seeds × 50 rollouts × ~8 s/rollout). Total suite ≈ 4 h.

## Status of the 4 axes

| Axis | Wrapper | Configs | Status |
|---|---|---|---|
| **spatial** | `roboeval.envs.perturb.SpatialShiftWrapper` | `spatial/act_spatial_y{+,-}{1,3,5}cm.yaml` (6 cells) | ✅ implemented |
| visual | `_make_visual_wrapper` (NotImplementedError) | (planned) `visual/act_visual_lighting_{plus,minus}{30,60}pct.yaml`, `visual/act_visual_distractor.yaml` | scaffolded, not implemented |
| dynamic | `_make_dynamic_wrapper` (NotImplementedError) | (planned) `dynamic/act_dynamic_push_at_{25,50,75}pct.yaml` | scaffolded, not implemented |
| **temporal** | `roboeval.envs.perturb.TemporalDelayWrapper` | `temporal/act_temporal_delay_{1,3,5}steps.yaml` (3 cells) | ✅ implemented |

## Planned cells (PRD §6.4)

| Axis | Cell | `perturbation:` block |
|---|---|---|
| spatial | y±1, ±3, ±5 cm | `kind: spatial, dx_m: 0.0, dy_m: ±0.01/0.03/0.05` |
| visual | lighting ±30%, ±60% | `kind: visual, lighting_scale: 0.7/0.4/1.3/1.6` |
| visual | distractor | `kind: visual, distractor: true` |
| dynamic | push at 25/50/75% | `kind: dynamic, push_dx_m: 0.02, push_at_fraction: 0.25/0.5/0.75` |
| temporal | delay 1/3/5 steps | `kind: temporal, delay_steps: 1/3/5` |

Total ≈ 24 cells × 150 rollouts = 3,600 rollouts. The PRD §6.4 estimate of "~1,800 rollouts" assumes each axis has ~3 intensities; this layout has more cells per axis to make the degradation curves smoother. Trim before running if compute is tight.

## Adding a new cell

1. Drop a YAML in the appropriate axis dir with `extends: configs/baseline/act_nominal.yaml` at the top.
2. Override only the `perturbation:` block + the `wandb.name_prefix` / `wandb.tags`.
3. The eval CLI auto-merges the parent's policy/env/eval/success blocks; nothing else changes.

## How `extends:` works

`roboeval.evaluation.config.load_eval_config(path)` reads the YAML, resolves `extends:` recursively (max depth 8 to catch cycles), and merges via `OmegaConf.merge(parent, child)`. Child values override parent values; nested dicts merge recursively, lists replace wholesale. The `extends:` key is stripped from the returned config.

The `${calibration:...}` resolver continues to work because it's registered before `load_eval_config` runs in `_cmd_evaluate`.
