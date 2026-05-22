# Phase 5 — Honest-Null Blog Post (Design)

**Status:** approved 2026-05-21 · **Owner:** Rubeno Dechua · **Targets:** PRD §4.1 (primary audience: hiring teams), PRD §9 (Benchmark Report / blog post deliverable), PRD §9.1 (writeup quality checklist).

## 1. Goal

Ship one Markdown blog post (~2000 words) that tells the honest-null Phase 4 story to hiring managers at robot-learning companies and the wider ML community. The post is the *one* artifact that turns the repo from "an evaluation harness with results" into "a publishable engineering narrative." It links to the live dashboard for interactive data and the repo for reproducible code.

PRD §9.1 acceptance for the writeup deliverable:

- Methods section reproducible from description alone.
- Results section includes uncertainty estimates (mean ± std, N, seed count).
- Limitations section present and honest.
- Every figure has a caption with axis labels and units.

## 2. Audience

Primary: engineers and hiring managers at Physical Intelligence, Skild AI, 1X, Figure, Dexterity, Covariant, BD AI Institute, Apptronik, Agility, GM Autonomy, Waymo, Cruise (PRD §4.1).

Secondary: the robot-learning research community on Twitter / Hugging Face / GitHub.

Implied reader: skims for ~30 seconds, looks for headers and the TL;DR, decides whether to read more. Comfortable with PPO / Welch's t / standard RL terminology; not interested in re-derivations.

## 3. Format & length

- **Markdown** (`.md`), GitHub-native rendering. Mermaid blocks for one diagram.
- **~2000 words** (allow 1900–2100). Tight enough to read in 8–10 minutes; long enough to carry the diagnosis.
- **Three figures** embedded inline as PNG. Captions inline below each.
- **One mermaid diagram** for the residual architecture (no PNG dependency).
- **No code blocks longer than ~10 lines** — link to the repo for full source.

## 4. Hosting

- **Source of truth:** `docs/blog/2026-05-21-honest-null-residual.md` in this repo, committed and renderable on github.com.
- **Promotion:** top-level `README.md` adds a "Read the writeup" link next to the existing "Live Demo" badge.
- **Cross-posting (deferred):** Hugging Face Blog or personal blog are optional v1.1; not in scope here.
- **No mkdocs build** — the post stands alone as Markdown. When the MkDocs site lands (separate Phase 5 task), the post is picked up via mkdocs-material's standard blog plugin without any change to the source file.

## 5. Voice

First-person narrative. "I trained the residual." "I expected this to work." "It didn't, and here's why."

Register: Lil'Log / Karpathy / John Schulman writeups. Direct, opinionated, ownership-forward, willing to say "I was wrong." No academic passive voice.

Avoid: marketing tone, jokes, emoji, "we" when only one author was involved, third-person passive ("a residual was trained").

## 6. Hook

Null-result-forward. The opening lede frames the failure as the story:

> "I trained a residual RL policy on top of ACT to fix its single biggest failure mode. It made the policy 13 percentage points worse. Here's why, and what I'd try next."

The Phase 3 cross-axis taxonomy appears as motivation for *why* +5 cm was the residual's target. The fix path (v1.1 distillation-init etc.) frames the failure as informative rather than terminal.

## 7. Structural approach

Linear scientific narrative: TL;DR → setup → hypothesis → experiment → result → diagnosis → fixes → closing. Standard structure, easy to scan, lowest narrative risk at this wordcount.

## 8. Outline

Section word budgets sum to ~2030; final draft will tighten to ≤ 2000.

| # | Section | Words | Beat |
|---|---------|-------|------|
| 1 | **Lede** | ~100 | Two sentences. The 13 pp drop. "Here's why." |
| 2 | **TL;DR box** | ~80 | 4 numbers (A/B/C TSR + Welch p), headline finding, v1.1 fix teaser. Bullet list. |
| 3 | **Setup: ACT on AlohaTransferCube** | ~250 | What the base policy is, the task, why 2026 robot learning cares about this benchmark. Reference to LeRobot's published 0.83 TSR baseline. |
| 4 | **Why +5 cm? The Phase 3 finding** | ~300 | Spatial brittle (67 pp drop at -5 cm) vs temporal robust (11 pp at 5 steps). Recovery dominates both. +5 cm picked because single-mode failure → cleanest residual target. **Figure F1: side-by-side degradation curves.** |
| 5 | **The hypothesis** | ~200 | Frozen base + 6-d MLP residual + α=0.05 + PPO. Sparse vs shaped reward as a two-arm ablation. **Figure F3: residual-architecture mermaid diagram.** Why I expected it to work. |
| 6 | **The experiment** | ~150 | 3 seeds × 50 rollouts × 3 conditions = 450 rollouts. Welch's t-test. Link to live dashboard. |
| 7 | **Result: the residual hurt** | ~250 | -13.3 pp sparse, -10.7 pp shaped, both p > 0.9. **Figure F2: 3-condition failure-mode stacked bar.** Note the qualitative shift: Approach failures 7× under sparse reward (directional miscorrection, not just jitter). |
| 8 | **Why it hurt: diagnosis** | ~350 | The four levers (base / reward / α / init). The α=0.05 × σ=0.135 → ±0.007 trap compounding over 385 steps. PPO's mean drifted nonzero, no positive-reward bootstrap to anchor it. Sparse-reward dead zone. MLP starts random, takes hundreds of thousands of steps to learn "do nothing." |
| 9 | **v1.1 fixes + closing** | ~250 | Distillation-init (1-line ResidualMLP change, top priority). Co-trainable α (custom SB3 policy class). ACT-encoder features. Smaller perturbation cells. Closing: code is open, harness is reusable, dashboard is live, here's where to look. |

## 9. Figures

| # | Figure | Source | Section | Caption |
|---|--------|--------|---------|---------|
| F1 | Cross-axis degradation curves (spatial cm vs temporal steps, side-by-side, ±σ ribbons) | New script `scripts/render_blog_figures.py` reads `data/headline.json` v2 and emits a single PNG. Static, self-contained outside the live dashboard. | §4 | "ACT's mean TSR vs perturbation. Left: spatial cube shifts in cm; 67 pp drop at -5 cm. Right: action-chunk delays in env steps; only 11 pp drop at 5 steps. ±σ shaded; 3 seeds × 50 rollouts per cell." |
| F2 | Phase 4 failure-mode stacked bar (A_base / B_sparse / C_shaped at +5 cm) | Existing `docs/figures/phase4_ablation_failure_distribution.png` (commit `5789526`). No regen needed. | §7 | "Failure-mode distribution at +5 cm spatial. The residual under both reward shapings shrinks the success bucket and grows Recovery; Approach failures jump 7× under sparse reward. 150 rollouts per condition, 3 seeds × 50." |
| F3 | Residual-architecture diagram | Inline mermaid block. GitHub renders natively. | §5 | "Per-step composition: ACT's frozen action plus an MLP residual scaled by α=0.05. PPO learns δ_θ(o) to maximise the sparse or shaped reward." |

PRD §9.1 demands axis labels, units, and captions for every figure. Each caption above already encodes all three.

## 10. Drafting workflow

1. **Spec lands** (this doc), committed to `docs/superpowers/specs/`.
2. **Implementation plan** (via writing-plans skill) breaks the post into per-section drafting tasks. Each section = one commit so review is incremental.
3. **F1 render script** lands in its own task — `scripts/render_blog_figures.py` + a small smoke test that asserts the figure builds without raising.
4. **Section drafts** happen one at a time, in outline order.
5. **Final pass:** read-through for flow, tighten word count, sanity-check that every figure has a caption, that the methods description names every config and run_id needed for reproduction.
6. **README link** lands at the end.
7. **STATE.md update** marks the writeup deliverable closed.

No CI gates on prose. The existing CI (ruff / mypy / pytest) is unaffected.

## 11. Acceptance

The writeup ships when:

- [ ] Final post at `docs/blog/2026-05-21-honest-null-residual.md` is ≤ 2100 and ≥ 1800 words (target 2000).
- [ ] All three figures embed and render on github.com.
- [ ] Every figure has a one-sentence caption with axis labels and units (PRD §9.1).
- [ ] Methods section names every config path and run_id needed for reproduction (PRD §9.1 "reproducible from description alone").
- [ ] Results section includes mean ± std, N, and seed count for every reported number (PRD §9.1).
- [ ] A limitations paragraph appears (PRD §9.1).
- [ ] Top-level `README.md` links to the post.
- [ ] `docs/STATE.md` updated to reflect the writeup deliverable closed.

## 12. Out of scope

- arXiv-style PDF compilation (deferred; the Markdown source can be Pandoc-compiled later).
- Hugging Face Blog cross-post (deferred to v1.1).
- MkDocs static-site setup (separate Phase 5 task).
- Demo video script (the writeup is *not* the video script — the video reuses figures and headline numbers but has its own narrative).
- Per-rollout trajectory plots (the dashboard is the place for interactive drilldowns; the blog points readers there).
- v1.1 residual experiments (the post *describes* the fix path; running the experiments is its own future Phase 4.5).
