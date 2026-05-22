# When residual RL made ACT worse: a 13-point honest null on AlohaTransferCube

<!-- 2026-05-21 · Rubeno Dechua · ~2000 words ·
     Repo: https://github.com/RubenoDechua/roboeval -->

<!-- §1 Lede -->

I spent a month testing how a state-of-the-art imitation learning policy
breaks when you nudge the world. The single biggest failure mode was
**Recovery** — the policy sweeps past the cube without engaging — and at
+5 cm of spatial shift, it accounts for 59% of all rollouts. I built a
residual RL loop on top of the frozen base policy to try to recover those
rollouts. Across two reward shapings and three seeds per arm, the residual
moved the task-success rate **−13.3 pp** under sparse reward and **−10.7 pp**
under shaped. This is the writeup of why that happened and what I'd try next.

<!-- §2 TL;DR box -->

> **TL;DR.** Frozen ACT on the +5 cm spatial cell scored **0.320 mean TSR**
> (3 seeds × 50 rollouts). A PPO residual on top of it scored **0.187**
> with sparse reward (Δ = −13.3 pp; Welch t = −2.95, p_one-sided = 0.034
> for *significant decrease*) and **0.213** with shaped reward (Δ = −10.7 pp;
> t = −1.95, p = 0.062). The residual hurts under both reward shapings. The
> failure mode is diagnosable: PPO drifts off zero with no positive bootstrap,
> α is large enough to compound, and the MLP starts random. The v1.1 fix
> path is concrete — see the end.

## Setup: ACT on AlohaTransferCube
<!-- §3 -->

The base policy is **ACT** (Action Chunking with Transformers), trained on
human demonstrations of the bimanual ALOHA Transfer Cube task and published
by the LeRobot project as
`lerobot/act_aloha_sim_transfer_cube_human`. The model card reports a
task-success rate of 0.83 across 500 sequential seeds; my reproduction with
3 seeds × 50 rollouts lands at 0.80 ± 0.057, well within sampling noise.
Verified, frozen, treated as the ground truth.

The task: two 7-DoF arms pick up a small cube with the right gripper, lift,
transfer to the left gripper, and place it. The simulator is `gym_aloha`'s
MuJoCo build; observations are 14-DoF agent positions plus three RGB cameras;
actions are absolute joint targets in [-1, 1]. Each episode runs up to 400
steps and terminates on a same-step contact between the left gripper and the
cube once the cube is above a calibrated z-threshold.

I evaluate against a custom **geometric** task-success rate (`mean_tsr_custom`)
that thresholds on cube position rather than environment reward. The custom
criterion calibrates `target_xy` and `xy_tolerance` from 50 nominal rollouts
and freezes them at `data/calibration/transfer_cube_target_xy.json`. This
removes one degree of freedom from the success signal — there's no debate
about whether a near-miss counts.

Why a 2026 reader should care: ACT is the cheapest competent bimanual
manipulation policy published this year, every robot-learning lab in PRD §4.1
is either using or replacing it, and "what does it break on" is the question
underneath nearly every Robot Learning Engineer interview I've seen.

## Why +5 cm? What an evaluation harness told me
<!-- §4 — embeds docs/figures/cross_axis_degradation.png -->

Before reaching for residual RL, I built an evaluation harness with two
perturbation axes. **Spatial:** translate the cube's initial XY pose by
±1, ±3, ±5 cm. **Temporal:** delay the action chunk by 1, 3, or 5 env
steps. Same base policy, same 150 rollouts per cell, same geometric
success criterion. The story turned out to be asymmetric across axes:

![Cross-axis degradation](../figures/cross_axis_degradation.png)

*ACT's mean task-success rate vs perturbation magnitude. Left: spatial
cube shifts in cm; 67 pp drop at -5 cm vs nominal. Right: action-chunk
delays in env steps; only 11 pp drop at 5 steps. ±σ shaded across 3 seeds
× 50 rollouts per cell.*

**Spatial is brittle.** Pull the cube 5 cm in either direction and TSR
collapses; the curve is nonlinear and asymmetric, with the negative
direction degrading harder (12.7% at -5 cm versus 30.7% at +5 cm).
**Temporal is robust.** A 5-step delay only costs 11 pp — ACT's 100-step
action chunking absorbs latency that would wreck a stateless policy.

The failure-mode classifier I ran on every rollout (PRD §7.2 taxonomy:
Success / Grasp / Approach / Recovery / Oscillation / Timeout / Visual
confusion) gave a clearer signal still: under both axes the dominant
failure was **Recovery** — the gripper moves into roughly the right region
and then sweeps through without engaging. At +5 cm spatial that single
mode accounts for **59% of all rollouts**. One failure mode that the base
policy reliably reproduces is the cleanest possible target for a residual
correction; everywhere else on the perturbation grid the failures were
multi-modal and harder to attack. So: +5 cm spatial.

## The hypothesis: a small additive residual
<!-- §5 — embeds the mermaid diagram -->

## The experiment
<!-- §6 -->

## Result: the residual hurt the base
<!-- §7 — embeds docs/figures/phase4_ablation_failure_distribution.png -->

## Why it hurt: diagnosing the four levers
<!-- §8 -->

## What I'd try next (v1.1)
<!-- §9 — closes with code/dashboard/docs links -->
