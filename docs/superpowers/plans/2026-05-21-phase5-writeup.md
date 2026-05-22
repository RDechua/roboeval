# Phase 5 Honest-Null Writeup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single ~2000-word Markdown blog post at `docs/blog/2026-05-21-honest-null-residual.md` that tells the Phase 4 honest-null story (residual RL hurt ACT by 13 pp on +5 cm), backed by three figures and linked from the top-level README.

**Architecture:** 12 incremental tasks. One task creates the scaffold and figure-render script; each subsequent task drafts one section of the post (Lede → Closing) and commits separately so review is per-section. The render script reads `data/headline.json` v2 (already self-contained per the prior CI fix) and emits a single PNG; the existing `phase4_ablation_failure_distribution.png` is reused; the residual-architecture diagram is inline mermaid.

**Tech Stack:** Markdown, matplotlib (for the cross-axis figure script), mermaid (for the inline diagram). No new Python runtime deps — matplotlib is already a project dependency.

**Workflow gates per commit (from project conventions):** `ruff check`, `ruff format --check`, `mypy --strict roboeval`, `pytest -q`. The blog post itself is prose, so the gates only exercise the figure script and its tests. Commit author `Rubeno Dechua <rubenodechua123@gmail.com>`, no Claude trailers.

---

## File Structure

| Path | Purpose |
|---|---|
| `docs/blog/2026-05-21-honest-null-residual.md` | The post itself, drafted section by section |
| `scripts/render_blog_figures.py` | One-off renderer for the cross-axis degradation figure (F1); reads `data/headline.json` v2 |
| `tests/scripts/test_render_blog_figures.py` | Smoke + invariant tests for the renderer |
| `docs/figures/cross_axis_degradation.png` | Output of the renderer, committed |
| `README.md` | Add "Read the writeup" link next to the existing Live Demo badge |
| `docs/STATE.md` | Mark the writeup deliverable closed |

The Phase 4 stacked-bar figure (`docs/figures/phase4_ablation_failure_distribution.png`) already exists from commit `5789526`; no regeneration needed.

---

## Task 1: Scaffold the post + render the cross-axis figure

**Files:**
- Create: `docs/blog/2026-05-21-honest-null-residual.md` (skeleton only)
- Create: `scripts/render_blog_figures.py`
- Create: `tests/scripts/test_render_blog_figures.py`
- Create: `docs/figures/cross_axis_degradation.png`

- [ ] **Step 1: Write the failing test for the figure renderer**

```python
# tests/scripts/test_render_blog_figures.py
"""Smoke tests for scripts.render_blog_figures."""

from __future__ import annotations

from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

from scripts.render_blog_figures import render_cross_axis_degradation  # noqa: E402


def test_render_cross_axis_degradation_emits_png(tmp_path: Path) -> None:
    """The renderer writes a non-empty PNG to the target path."""
    repo_root = Path(__file__).resolve().parents[2]
    out_path = tmp_path / "cross_axis.png"
    render_cross_axis_degradation(
        headline_path=repo_root / "data" / "headline.json",
        out_path=out_path,
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 1024  # at least 1 KB
```

- [ ] **Step 2: Run to confirm fails**

```bash
.venv/bin/pytest tests/scripts/test_render_blog_figures.py -v
```

Expected: ImportError on `scripts.render_blog_figures`.

- [ ] **Step 3: Implement the renderer**

```python
# scripts/render_blog_figures.py
"""Render static figures for the Phase 5 blog post.

Reads ``data/headline.json`` v2 (the self-contained dashboard data
artifact) and emits PNG figures into ``docs/figures/``. The figures
mirror panels of the live dashboard but are baked at commit time so
the blog post stays self-contained when the dashboard Space is asleep.

Usage::

    python -m scripts.render_blog_figures
    # writes docs/figures/cross_axis_degradation.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

_LOG = logging.getLogger("scripts.render_blog_figures")

_PRIMARY_COLOR = "#2E86AB"
_RIBBON_COLOR = (46 / 255, 134 / 255, 171 / 255, 0.18)


def _load_cells(headline_path: Path) -> list[dict[str, object]]:
    payload = json.loads(headline_path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(
            f"render_blog_figures needs headline.json schema 2, "
            f"got {payload.get('schema_version')!r}"
        )
    cells = payload["cells"]
    assert isinstance(cells, list)
    return cells


def render_cross_axis_degradation(
    *, headline_path: Path, out_path: Path
) -> None:
    """Render the side-by-side spatial + temporal degradation panels.

    Args:
        headline_path: Path to ``data/headline.json`` (schema v2).
        out_path: PNG output path. Parent directory is created if absent.
    """
    cells = _load_cells(headline_path)

    spatial = sorted(
        [c for c in cells if c["axis"] in ("spatial", "nominal")],
        key=lambda c: float(c["magnitude"]),  # type: ignore[arg-type]
    )
    temporal = sorted(
        [c for c in cells if c["axis"] in ("temporal", "nominal")],
        key=lambda c: float(c["magnitude"]),  # type: ignore[arg-type]
    )

    spatial_x = [float(c["magnitude"]) * 100.0 for c in spatial]  # m -> cm
    spatial_y = [float(c["mean_tsr_custom"]) for c in spatial]
    spatial_sigma = [float(c["std_tsr_custom"]) for c in spatial]

    temporal_x = [float(c["magnitude"]) for c in temporal]
    temporal_y = [float(c["mean_tsr_custom"]) for c in temporal]
    temporal_sigma = [float(c["std_tsr_custom"]) for c in temporal]

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True
    )

    for ax, xs, ys, sigmas, x_label in (
        (ax_l, spatial_x, spatial_y, spatial_sigma, "Cube shift (cm)"),
        (ax_r, temporal_x, temporal_y, temporal_sigma, "Action delay (env steps)"),
    ):
        upper = [y + s for y, s in zip(ys, sigmas, strict=True)]
        lower = [y - s for y, s in zip(ys, sigmas, strict=True)]
        ax.fill_between(xs, lower, upper, color=_RIBBON_COLOR, linewidth=0)
        ax.plot(
            xs, ys, color=_PRIMARY_COLOR, linewidth=2.4, marker="o", markersize=6
        )
        ax.axvline(0, color="#888", linestyle=":", linewidth=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.25)

    ax_l.set_ylabel("Mean task-success rate")
    ax_l.set_title("Spatial perturbation")
    ax_r.set_title("Temporal delay")
    fig.suptitle(
        "ACT on AlohaTransferCube — degradation across perturbation axes",
        fontsize=12,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("wrote %s", out_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    render_cross_axis_degradation(
        headline_path=repo_root / "data" / "headline.json",
        out_path=repo_root / "docs" / "figures" / "cross_axis_degradation.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the renderer once to produce the committed PNG**

```bash
.venv/bin/python -m scripts.render_blog_figures
```

Expected: `INFO: wrote .../docs/figures/cross_axis_degradation.png`. Inspect the PNG manually before continuing.

- [ ] **Step 5: Write the post skeleton**

```markdown
# When residual RL made ACT worse: a 13-point honest null on AlohaTransferCube

<!-- 2026-05-21 · Rubeno Dechua · ~2000 words ·
     Repo: https://github.com/RubenoDechua/roboeval -->

<!-- §1 Lede goes here -->

<!-- §2 TL;DR box -->

## Setup: ACT on AlohaTransferCube
<!-- §3 -->

## Why +5 cm? What an evaluation harness told me
<!-- §4 — embeds docs/figures/cross_axis_degradation.png -->

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
```

- [ ] **Step 6: Run gates**

```bash
.venv/bin/ruff check scripts tests/scripts
.venv/bin/ruff format --check scripts tests/scripts
.venv/bin/mypy --strict roboeval
.venv/bin/pytest tests/scripts/test_render_blog_figures.py -v
```

Expected: all green; one test passes.

- [ ] **Step 7: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md \
        scripts/render_blog_figures.py \
        tests/scripts/test_render_blog_figures.py \
        docs/figures/cross_axis_degradation.png
git commit -m "feat(writeup): scaffold blog post + cross-axis figure renderer"
git push origin main
```

---

## Task 2: Draft §1 Lede (~100 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (replace `<!-- §1 Lede goes here -->`)

- [ ] **Step 1: Replace the lede placeholder with the drafted lede**

```markdown
<!-- §1 Lede -->

I spent a month testing how a state-of-the-art imitation learning policy
breaks when you nudge the world. The single biggest failure mode was
**Recovery** — the policy sweeps past the cube without engaging — and at
+5 cm of spatial shift, it accounts for 59% of all rollouts. I built a
residual RL loop on top of the frozen base policy to try to recover those
rollouts. Across two reward shapings and three seeds per arm, the residual
moved the task-success rate **−13.3 pp** under sparse reward and **−10.7 pp**
under shaped. This is the writeup of why that happened and what I'd try next.
```

- [ ] **Step 2: Verify word count in the lede block**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'<!-- §1 Lede -->(.+?)<!-- §2', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: between 90 and 130 words.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §1 lede"
git push origin main
```

---

## Task 3: Draft §2 TL;DR (~80 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (replace `<!-- §2 TL;DR box -->`)

- [ ] **Step 1: Replace the TL;DR placeholder**

```markdown
<!-- §2 TL;DR box -->

> **TL;DR.** Frozen ACT on the +5 cm spatial cell scored **0.320 mean TSR**
> (3 seeds × 50 rollouts). A PPO residual on top of it scored **0.187**
> with sparse reward (Δ = −13.3 pp; Welch t = −2.95, p_one-sided = 0.034
> for *significant decrease*) and **0.213** with shaped reward (Δ = −10.7 pp;
> t = −1.95, p = 0.062). The residual hurts under both reward shapings. The
> failure mode is diagnosable: PPO drifts off zero with no positive bootstrap,
> α is large enough to compound, and the MLP starts random. The v1.1 fix
> path is concrete — see the end.
```

- [ ] **Step 2: Sanity-check the numbers against the data**

```bash
.venv/bin/python -c "
import json
d = json.load(open('data/headline.json'))
for a in d['ablation']:
    print(a['condition_id'], a['label'], 'mean=', a['mean_tsr_custom'])
for w in d['welch_tests']:
    print('Welch', w['arm_id'], 't=', w['t_statistic'], 'p=', w['p_one_sided'])
"
```

Expected: A=0.32, B=0.187, C=0.213. Welch arms B and C present.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §2 TL;DR"
git push origin main
```

---

## Task 4: Draft §3 Setup (~250 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (under `## Setup: ACT on AlohaTransferCube`)

- [ ] **Step 1: Replace the section body**

```markdown
## Setup: ACT on AlohaTransferCube

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
```

- [ ] **Step 2: Word-count check**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'## Setup:(.+?)## Why', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: 230–280 words.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §3 setup"
git push origin main
```

---

## Task 5: Draft §4 Why +5 cm? (~300 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (under `## Why +5 cm? What an evaluation harness told me`)

- [ ] **Step 1: Replace the section body, embedding F1**

```markdown
## Why +5 cm? What an evaluation harness told me

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
```

- [ ] **Step 2: Word-count check**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'## Why \+5 cm(.+?)## The hypothesis', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: 270–330 words.

- [ ] **Step 3: Verify the figure renders in GitHub markdown**

Push the commit (Step 4) and open the post on github.com; confirm the PNG appears. If broken, the issue is the relative path `../figures/cross_axis_degradation.png` — adjust to `../../docs/figures/...` if needed (`docs/blog/<post>.md` is 1 level deeper than `docs/figures/`).

- [ ] **Step 4: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §4 cross-axis findings + figure F1"
git push origin main
```

---

## Task 6: Draft §5 Hypothesis (~200 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (under `## The hypothesis: a small additive residual`)

- [ ] **Step 1: Replace the section body, embedding the mermaid diagram (F3)**

````markdown
## The hypothesis: a small additive residual

```mermaid
flowchart LR
    O[observation o_t] --> ACT[Frozen ACT base]
    O --> R[Residual MLP δ_θ(o_t)]
    ACT --> S[a_base]
    R --> SCALE[× α = 0.05]
    S --> SUM((+))
    SCALE --> SUM
    SUM --> A[a_t = a_base + α · δ_θ]
```

*Per-step composition: ACT's frozen action plus an MLP residual scaled
by α = 0.05. PPO learns δ_θ to maximize the reward.*

The architecture is small on purpose. The residual is a two-layer GELU MLP
(256 → 256) emitting a 6-d correction in joint-target space, scaled by
α=0.05. With ACT's per-dim action std around σ=0.135, that bounds the
residual's per-step perturbation at roughly ±0.007 — small enough, I
thought, that it would either help or do nothing.

I ran two reward shapings as a paired ablation. **Sparse:** +1 on the
geometric success criterion firing, 0 otherwise. **Shaped:** sparse plus a
distance-shaping term `-w · ‖cube_xy - target_xy‖₂` with `w = 1.0`,
matching the PRD §8 design. The intent was to check whether sparse reward
was the limiting factor; the alternative hypothesis was that the residual
just lacks signal.
````

- [ ] **Step 2: Word-count check**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'## The hypothesis(.+?)## The experiment', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: 180–230 words.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §5 hypothesis + residual architecture diagram (F3)"
git push origin main
```

---

## Task 7: Draft §6 The experiment (~150 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (under `## The experiment`)

- [ ] **Step 1: Replace the section body**

```markdown
## The experiment

Three conditions × 3 seeds × 50 rollouts = **450 rollouts** total.

- **A** (control): frozen ACT only, no residual. `act_spatial_y+5cm.yaml`. W&B run `w6k2wole`.
- **B** (sparse): frozen ACT + residual PPO trained for ~500 k env steps
  with the sparse reward. `residual_ppo_y+5cm_sparse.yaml`. Eval run
  `o6ukyo53`.
- **C** (shaped): same architecture and training budget, shaped reward.
  `residual_ppo_y+5cm_shaped.yaml`. Eval run `43czuigy`.

I report mean ± std of `mean_tsr_custom` across the three seed groups for
each arm, plus a one-sided Welch's t-test comparing each residual arm to
condition A (null hypothesis: arm ≤ A; alternative: arm > A; rejection
means the residual *helps*). I also report the per-rollout failure-mode
distribution from the PRD §7.2 classifier. All raw evidence — `eval_results_*.json`,
`auto_labels_*.json`, the aggregator output — is committed under
`outputs/` and `data/taxonomy/`, and the live dashboard at
`huggingface.co/spaces/RubenoDechua/roboeval` exposes the same numbers
interactively.
```

- [ ] **Step 2: Word-count check**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'## The experiment(.+?)## Result', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: 130–180 words.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §6 experiment design + run IDs"
git push origin main
```

---

## Task 8: Draft §7 Result (~250 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (under `## Result: the residual hurt the base`)

- [ ] **Step 1: Replace the section body, embedding F2**

```markdown
## Result: the residual hurt the base

The headline numbers (mean ± std across 3 seed groups, n=50 each):

| Arm | Mean TSR | Δ vs A | Welch's t | p (one-sided, residual > A) |
|---|---|---|---|---|
| A — frozen base | 0.320 ± 0.059 | — | — | — |
| B — residual, sparse | 0.187 ± 0.025 | **−13.3 pp** | −2.95 | 0.966 |
| C — residual, shaped | 0.213 ± 0.050 | **−10.7 pp** | −1.95 | 0.938 |

Both residual arms fall below the base. The one-sided p-values are large
because the alternative was "the residual helps"; flip the direction and
the residual's negative effect is significant at p ≈ 0.03 for sparse and
borderline (p ≈ 0.06) for shaped. The per-seed spread is tight (0.025 for
sparse) — this is real signal, not noise across runs.

The failure-mode distribution is the more informative half of the result:

![Failure modes at +5 cm](../figures/phase4_ablation_failure_distribution.png)

*Failure-mode distribution at +5 cm spatial. The residual under both
reward shapings shrinks the success bucket and grows Recovery; Approach
failures jump 7× under sparse reward. 150 rollouts per condition,
3 seeds × 50.*

Three qualitative shifts stand out. **Success** collapses from 30.7%
(base) to 18.7% (sparse) / 21.3% (shaped). **Recovery** grows from 59.3%
to 70.7% / 73.3% — the residual is making the dominant failure *more*
dominant. And **Approach failures** jump from 0.7% (base) to 5.3%
(sparse) — a 7× increase. The residual is not just adding jitter; it is
occasionally yanking the gripper away from the cube. That's a directional
miscorrection, not noise. The next section asks why.
```

- [ ] **Step 2: Word-count check**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'## Result(.+?)## Why it hurt', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: 220–290 words.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §7 result + figure F2"
git push origin main
```

---

## Task 9: Draft §8 Diagnosis (~350 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (under `## Why it hurt: diagnosing the four levers`)

- [ ] **Step 1: Replace the section body**

```markdown
## Why it hurt: diagnosing the four levers

Residual RL on a frozen base policy has four design knobs: the base, the
reward, the blend coefficient α, and the residual network's initialization.
Reasoning through each:

**The base.** ACT at +5 cm scored 30.7% — already in the long-tail regime
where 59% of rollouts are Recovery. There isn't much *headroom* for a
small additive correction here; the cube is far enough from the trained
distribution that the base's joint targets are pointing at the wrong
position to begin with. A small residual cannot fix a large pose error;
it can only nudge a near-miss into a hit. **The base policy was probably
too broken at this cell for a same-architecture additive fix.**

**The reward.** The sparse reward is +1 only on the success criterion
firing — and at this cell, only 19% of *the residual's own training
rollouts* fired the criterion. That's a sparse-reward dead zone:
PPO's advantage estimates are dominated by terminal value, the policy
gradient is mostly noise, and the entropy bonus drifts the mean
nonzero. The shaped reward was supposed to fix that, and it does narrow
the gap (−10.7 pp vs. −13.3 pp), but not close it. **Reward shaping
helped a little; it didn't get over the hump.**

**Alpha.** α=0.05 × σ=0.135 gives a per-dim perturbation of about ±0.007.
That's small per step but compounds: ACT runs ~385 steps to success and
the residual integrates an action perturbation every one of them. If
PPO's mean has drifted by even 0.05 of a standard deviation in a
*consistent direction*, the gripper accumulates several centimeters of
trajectory error before the episode ends. The Approach-failure jump under
sparse reward is exactly what this looks like geometrically.

**The init.** The residual MLP starts with default He-uniform weights and
a learnable log-σ initialized at -2 (per SB3 defaults). At t=0 the
network's output is small-but-random, not zero. PPO has to learn "do
nothing" from scratch, in a sparse-reward regime, which is harder than
learning a correction. **The residual never gets to "no harm" as a
lower bound** — and "no harm" is what α should buy us in principle.

Three of those four levers have concrete v1.1 fixes. Below.
```

- [ ] **Step 2: Word-count check**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'## Why it hurt(.+?)## What I', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: 320–400 words. If the section runs over by 30+ words, tighten the third paragraph (Alpha) first — that's the densest of the four and easiest to compress.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §8 diagnosis of the four design levers"
git push origin main
```

---

## Task 10: Draft §9 What I'd try next (~250 words)

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (under `## What I'd try next (v1.1)`)

- [ ] **Step 1: Replace the section body**

```markdown
## What I'd try next (v1.1)

The diagnosis points at four concrete fixes, ordered by leverage:

**Distillation-init residual** (top priority, smallest change). Zero the
output-layer weights and bias of `ResidualMLP` and shrink the rest by 0.1
at construction time. The policy now starts as "do nothing" — α · δ is
~0 for the first thousands of PPO steps, so the base policy's performance
is a strict lower bound. This is a one-line `nn.init.zeros_` change in
`roboeval/residual/policy.py`. It directly addresses lever 4.

**Co-trainable α.** Move α out of the YAML and into the residual policy
as a `nn.Parameter` with a small initial value (0.01) and an `L2`
penalty. PPO learns when to use the residual rather than always
saturating it. This addresses lever 3 and pairs naturally with the
distillation init — together they bound the residual's worst case to
"no harm."

**ACT-encoder features.** The current residual reads raw 14-DoF agent
positions. Wire the residual's input through ACT's transformer encoder
(via the `feature_extractor` slot the codebase already has) so it sees
the same vision-conditioned representation the base does. This should
make the residual sim-to-real portable and addresses lever 1 by sharing
the base's perceptual prior.

**Smaller perturbation cells.** Re-run the ablation at +1 cm and +3 cm
where the base still has 72% and 55% TSR respectively. There's actual
headroom there for an additive correction; if the residual *still* hurts
in that regime, the architectural story above is wrong and we have
something interesting to chase.

The eval harness, the failure-mode classifier, the residual training
loop, the aggregator, and the dashboard are all in place to run any of
these in an afternoon. The code is at
[github.com/RubenoDechua/roboeval](https://github.com/RubenoDechua/roboeval);
the live dashboard is at
[huggingface.co/spaces/RubenoDechua/roboeval](https://huggingface.co/spaces/RubenoDechua/roboeval).
The PRD is in `docs/PRD.md`, the per-week research log is in
`docs/research-log.md`, and the per-condition writeup (with the full
Welch's t-test pipeline) is in `docs/phase4_ablation.md`.

If you're hiring for evaluation engineering or residual RL and want to
talk about this, my email is in the GitHub profile.
```

- [ ] **Step 2: Word-count check**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
m = re.search(r'## What I.+?\n(.+)$', text, re.S)
body = re.sub(r'\s+', ' ', m.group(1)).strip()
print('words:', len(body.split()))
"
```

Expected: 280–360 words. The section is allowed to run slightly long because the closing email/contact line is part of it.

- [ ] **Step 3: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): §9 v1.1 fix path + closing"
git push origin main
```

---

## Task 11: Final pass — limitations, word-count tighten, end-to-end read

**Files:**
- Modify: `docs/blog/2026-05-21-honest-null-residual.md` (insert a Limitations paragraph; tighten any section that bust its budget)

- [ ] **Step 1: Insert a Limitations paragraph before §9 ("What I'd try next")**

This is a PRD §9.1 hard requirement. Add this block as the last paragraph
of §8 ("Why it hurt") — same section, separated by a blank line — so the
flow stays linear:

```markdown
### Limitations

Three honest caveats. **(1) Single base policy.** I evaluated only ACT;
diffusion-policy results may differ both quantitatively and in failure
morphology. **(2) Single cell.** The ablation ran at +5 cm spatial only.
The architectural fixes above need re-running at +1 and +3 cm where the
base has more headroom before we can claim they generalise. **(3) Small
N.** 3 seeds × 50 rollouts per arm is enough for the qualitative story
but the Welch's t p-values are wide because df ≈ 3; a follow-up should
run 5+ seed groups to tighten the confidence intervals.
```

- [ ] **Step 2: Re-run all section word counts**

```bash
.venv/bin/python -c "
import re
text = open('docs/blog/2026-05-21-honest-null-residual.md').read()
# Strip front matter and code fences for the count.
prose = re.sub(r'\`\`\`.+?\`\`\`', '', text, flags=re.S)
prose = re.sub(r'<!--.+?-->', '', prose, flags=re.S)
words = len(re.findall(r'\b[\w\-]+\b', prose))
print('total words (excl code blocks and comments):', words)
"
```

Expected: between 1800 and 2100. If over 2100, tighten §8 first (densest section); then §3 (highest narrative slack).

- [ ] **Step 3: Read the post end-to-end on github.com**

Push the commit (Step 4) and open `https://github.com/RubenoDechua/roboeval/blob/main/docs/blog/2026-05-21-honest-null-residual.md` in a browser.

Manual checklist:

- [ ] Both PNG figures render.
- [ ] Mermaid diagram renders.
- [ ] Headers form a sensible TOC.
- [ ] No broken internal links.
- [ ] Tone is consistent (first-person, direct).

If any check fails, fix in this same commit before continuing.

- [ ] **Step 4: Commit**

```bash
git add docs/blog/2026-05-21-honest-null-residual.md
git commit -m "feat(writeup): limitations + final read-through tightening"
git push origin main
```

---

## Task 12: README link + STATE.md closure

**Files:**
- Modify: `README.md`
- Modify: `docs/STATE.md`

- [ ] **Step 1: Add the "Read the writeup" link to the top-level README**

Open `README.md` and insert the writeup link immediately after the existing
Live Demo badge:

```markdown
[![Live Demo](https://img.shields.io/badge/HF%20Spaces-Live%20Demo-blue?logo=huggingface)](https://huggingface.co/spaces/RubenoDechua/roboeval)
[![Writeup](https://img.shields.io/badge/Blog-Honest%20null-blueviolet)](docs/blog/2026-05-21-honest-null-residual.md)
```

- [ ] **Step 2: Update `docs/STATE.md` to mark the writeup closed**

Find the "Next session intent" list and replace item 3 (the blog post)
with a "closed" entry. Concretely, change this block:

```markdown
3. **Blog post / arXiv-style writeup** — builds on
   `docs/phase4_ablation.md` + the Phase 3 cross-axis findings in
   `docs/research-log.md`. Honest-null framing is the hook.
```

into:

```markdown
3. **Blog post landed** ✓ — `docs/blog/2026-05-21-honest-null-residual.md`,
   ~2000 words, honest-null framing, three figures (cross-axis curves,
   Phase 4 stacked bar, mermaid residual architecture). Linked from
   README. arXiv-style PDF cross-post deferred.
```

- [ ] **Step 3: Run all gates one final time**

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy --strict roboeval
.venv/bin/pytest -q
```

Expected: all green. Prose changes shouldn't break gates, but verify.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/STATE.md
git commit -m "docs(phase5): link writeup from README + close STATE.md item"
git push origin main
```

---

## Self-Review

**Spec coverage (against `2026-05-21-phase5-writeup-design.md`):**

| Spec section | Plan task(s) |
|---|---|
| §1 Goal | Task 1–12 collectively |
| §2 Audience | §5 (Hypothesis) + §9 (Closing) explicit voice; not a separate task |
| §3 Format & length | Word-count checks at each section task; Task 11 final total |
| §4 Hosting (`docs/blog/...md`) | Task 1 scaffold |
| §5 Voice (first-person) | Embedded in every draft step's prose |
| §6 Hook (null-result-forward) | Task 2 (§1 Lede) |
| §7 Structural approach (linear) | Section order across Tasks 2–10 |
| §8 Outline (9 sections, word budgets) | Tasks 2–10 each match one section |
| §9 Figures (F1, F2, F3 + captions) | F1 in Task 1+5, F2 in Task 8, F3 in Task 6; captions in same tasks |
| §10 Drafting workflow | Task 1 scaffold → Tasks 2–10 per-section → Task 11 final pass → Task 12 promotion |
| §11 Acceptance | Task 11 (limitations + word-count + figure check) + Task 12 (README + STATE) |
| §12 Out of scope | No task; correctly absent |

**Placeholder scan:** none. Every draft step contains the full prose body to commit.

**Type consistency:** the renderer (`render_cross_axis_degradation`) and its test (`test_render_cross_axis_degradation_emits_png`) match between Task 1 and any later reference.

**Risks noted, not blocking:**
- The relative image path `../figures/...` in §4 and §7 assumes the post lives at `docs/blog/<post>.md` and figures live at `docs/figures/`. The check in Task 5 Step 3 catches this empirically on github.com.
- Mermaid render compatibility on github.com is solid as of 2024-05; HF Blog cross-posting (deferred) may need a separate path.
- The Welch's t-test interpretation in §7 (Task 8) flips signs deliberately — read the table carefully when reviewing. If a future reader gets confused, the §7 caption explains the one-sided convention.
