# κ Relabel Runbook — Closing G3 on 2026-05-24

> **Status:** waiting on embargo unlock. Don't read the auto labels before
> filling in `manual_failure_mode` — that's the whole point of the embargo
> (PRD §7.3).

**Unlock at:** `2026-05-24T22:01:39 UTC` ≈ Sunday **3:01 PM Pacific / 6:01 PM Eastern**.

**Acceptance (PRD §7.3):** Cohen's κ > 0.6 across the combined 36-rollout sample.
The scoring script enforces this.

---

## Inputs (committed today)

| File | Content |
|---|---|
| `data/taxonomy/relabel_sample_18xb5ob0.json` | 18 rollouts from the **-5 cm** cell, `manual_failure_mode: null` |
| `data/taxonomy/relabel_sample_alr0r0p2.json` | 18 rollouts from the **+5 cm** cell, `manual_failure_mode: null` |
| `data/taxonomy/auto_labels_18xb5ob0.json` | Auto-classifier output for run `18xb5ob0` (not to be opened before scoring) |
| `data/taxonomy/auto_labels_alr0r0p2.json` | Auto-classifier output for run `alr0r0p2` |
| `scripts/relabel_score.py` | Embargo-checking κ scorer; will refuse to run before unlock |

The auto label files are gitignored locally; if any are missing on Sunday, regenerate
with `scripts/relabel_from_wandb.py <run_id>`.

---

## Sunday's 5 steps

### Step 1 — Verify it's after the unlock

```bash
date -u
```

The printed timestamp must be ≥ `2026-05-24T22:01:39 UTC`. If you're earlier, wait.

### Step 2 — Watch each rollout

Open the two W&B runs in separate tabs:

- **-5 cm cell:** https://wandb.ai/rdechua-university-of-san-francisco/roboeval/runs/18xb5ob0
- **+5 cm cell:** https://wandb.ai/rdechua-university-of-san-francisco/roboeval/runs/alr0r0p2

For each row in the two sample JSONs, find that rollout's video by
`seed_group` + `rollout_idx` (W&B logs them as a per-rollout table) and watch
it once. Classify by *what you see*, not by what you remember the auto
classifier saying weeks ago.

### Step 3 — Fill in the labels

Open each sample JSON in your editor:

```bash
code data/taxonomy/relabel_sample_18xb5ob0.json
code data/taxonomy/relabel_sample_alr0r0p2.json
```

For every entry, change `"manual_failure_mode": null` to one of:

| Label | When it applies (PRD §7.2) |
|---|---|
| `success` | Cube reaches the target placement zone |
| `grasp_failure` | Gripper closed on cube but cube slipped before lift |
| `approach_failure` | Gripper never touched the cube (off-trajectory) |
| `recovery_failure` | Gripper passed over the cube without engaging (the dominant +5 cm failure) |
| `action_oscillation` | Visible high-frequency action chatter, no progress |
| `timeout` | Episode hit max steps with no clear progress and no contact |
| `visual_confusion` | Policy went after the wrong object / phantom target |
| `needs_review` | Honestly cannot decide — leave as the last resort, not the default |

The categories are mutually exclusive. Pick *one* per rollout.

**Don't open the `auto_labels_*.json` files yet.** They're the comparison rater; reading them invalidates the blinding.

### Step 4 — Score

From the repo root:

```bash
.venv/bin/python -m scripts.relabel_score \
    data/taxonomy/relabel_sample_18xb5ob0.json \
    data/taxonomy/relabel_sample_alr0r0p2.json
```

Expected output (success case):

```
=== 18xb5ob0 (relabel_sample_18xb5ob0.json) ===
  rollouts scored        : 18
  observed agreement     : 0.8889
  expected (chance) agree: 0.3210
  Cohen's kappa          : 0.8364
  verdict: PASS (κ > 0.6)

=== alr0r0p2 (relabel_sample_alr0r0p2.json) ===
  rollouts scored        : 18
  observed agreement     : 0.8333
  expected (chance) agree: 0.2967
  Cohen's kappa          : 0.7634
  verdict: PASS (κ > 0.6)

=== Combined verdict ===
  samples scored: 2
  per-sample κ  : [0.8364, 0.7634]
  ALL PASS — G3 (PRD §7.3) closes.
```

If `verdict: FAIL` on either cell:

1. The script prints per-sample κ; the lower one is where the disagreement concentrates.
2. Open the corresponding `auto_labels_<run_id>.json` and diff which categories you and the auto-classifier disagreed on.
3. Decide whether (a) your labels were inconsistent and you want to re-watch, or (b) the classifier has a real bug. If (b), file a follow-up issue — don't try to "fix" κ post hoc.

### Step 5 — Close G3 in STATE.md

If PASS, edit `docs/STATE.md`'s "Quality gates" section to mark G3 closed. Replace whatever the current "G3 — Robustness & Taxonomy" line says with something like:

```markdown
| **G3 — Robustness & Taxonomy** | End of Phase 3 (Week 5) | **CLOSED 2026-05-24.**
Cohen's κ = <combined value> on 36 manually-relabelled rollouts (18 per cell at
±5 cm), well above the PRD §7.3 floor of 0.6. Auto-classifier validated. |
```

Then commit + push:

```bash
git add data/taxonomy/relabel_sample_*.json docs/STATE.md
git commit -m "feat(taxonomy): G3 closed — Cohen's kappa <value> on 36-rollout blind relabel"
git push origin main
```

The `relabel_sample_*.json` files are tracked (they're embargo artifacts, not the gitignored auto labels), so committing the filled-in manual labels is the right move.

---

## What I'll do after you close G3

Ping me with the κ values and I'll:

1. Add a one-paragraph note to `docs/research-log.md` under a "Week 7" heading.
2. Update STATE.md's "Phase" block to mark G3 done.
3. Add a sentence to the blog post's §6 "The experiment" tightening the taxonomy validation claim.

Estimated total time on Sunday: **~45 min** (30 min watching rollouts, 5 min editing JSON, 10 min scoring + commit).
