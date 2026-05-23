# Demo Video — Script, Shot List, and Recording Recipe

**Target:** 90-second narrated video, 1920×1080 minimum, MP4 (H.264).
**Acceptance (PRD §9.1):** narrated · failure modes visually annotated · ≤ 90 s · 1080p min.

Five beats, ~225 words at 2.5 words/sec. Time-stamps below assume the cuts hit ±0.5 s.

---

## Beat 1 · Hook (0:00 → 0:10, ~25 words)

**Narration:**

> ACT is a state-of-the-art imitation policy on the bimanual ALOHA Transfer Cube. At nominal,
> it succeeds 80% of the time. Shift the cube five centimeters, and it sweeps right past it.

**Visual:** Side-by-side picture-in-picture.
- **Left:** A1 — nominal rollout, successful pick-and-place. Mute audio. Loop or freeze on final placement.
- **Right:** A2 — +5 cm Recovery failure. Gripper closes on empty air past the cube. Loop or freeze on the miss.
- **On-screen text** at 0:08 (bottom-third lower-third): "+5 cm spatial shift → 59% Recovery failures".

**Cut at 0:10** to full-screen B1 (failure-mode bar chart).

---

## Beat 2 · The taxonomy (0:10 → 0:25, ~35 words)

**Narration:**

> I classified 150 rollouts per cell into six failure categories. Recovery — the gripper passing
> over the cube without engaging — dominates. It accounts for 59% of failures at +5 cm and
> emerges on both spatial and temporal perturbation axes.

**Visual:**
- **0:10 → 0:18:** Full-screen B1 — `docs/figures/spatial_failure_distribution.png`. Zoom slowly from full chart to the +5 cm column.
- **0:18 → 0:25:** Cross-cut to B2 — `docs/figures/cross_axis_degradation.png`. Animate the left (spatial) panel fading in first, then the right (temporal) panel.
- **On-screen text** at 0:20: "spatial brittle · temporal robust".

**Cut at 0:25** to full-screen C1.

---

## Beat 3 · The fix attempt (0:25 → 0:45, ~50 words)

**Narration:**

> I trained a PPO residual on top of the frozen ACT base — a small MLP correction scaled by
> alpha-equals-zero-point-zero-five. I ran two arms: a sparse success-only reward, and a
> distance-shaped reward. Three seeds, fifty rollouts each, four-hundred-fifty rollouts total.

**Visual:**
- **0:25 → 0:35:** Full-screen C1 — residual architecture diagram (mermaid render). Highlight the `α = 0.05` and `δ_θ(o)` symbols.
- **0:35 → 0:45:** Animate a small label block fading in:
  - `A frozen base only`
  - `B residual + sparse reward`
  - `C residual + shaped reward`
  - `3 seeds × 50 rollouts × 3 conditions = 450`.

**Cut at 0:45** to full-screen B3.

---

## Beat 4 · The result (0:45 → 1:05, ~50 words)

**Narration:**

> The residual hurt the base. Sparse reward dropped task success by 13.3 percentage points,
> shaped reward by 10.7. Both arms grew Recovery failures and grew Approach failures. The
> residual was occasionally yanking the gripper away from the cube — a directional miscorrection,
> not added noise.

**Visual:**
- **0:45 → 1:00:** Full-screen B3 — `docs/figures/phase4_ablation_failure_distribution.png`.
- **On-screen text** overlay at 0:50, large and centred over a translucent backdrop:
  - `A = 0.320`
  - `B = 0.187  (Δ −13.3 pp)`
  - `C = 0.213  (Δ −10.7 pp)`
- **0:55 → 1:05:** Highlight the Recovery and Approach bars with an arrow + circle annotation.

**Cut at 1:05** to D1.

---

## Beat 5 · Outro (1:05 → 1:25, ~35 words)

**Narration:**

> The repo ships the eval harness, the residual loop, an interactive dashboard, and a writeup
> explaining the four design knobs that need fixing. The honest-null is the result; the
> diagnosis is the contribution.

**Visual:**
- **1:05 → 1:15:** Screen recording D1 — interact with the live dashboard at `huggingface.co/spaces/rubenodechua/roboeval`. Toggle axis filter, change cell selector, click a failure-mode legend item.
- **1:15 → 1:25:** Card with three URLs, large text:
  - 🌐 `rubenodechua-roboeval.hf.space`  *(dashboard)*
  - ✍️ `github.com/RDechua/roboeval#blog`  *(writeup)*
  - 💻 `github.com/RDechua/roboeval`  *(code)*

**Hard cut to black** at 1:25.

---

## Shot list (recording checklist)

| ID | What | Where it comes from | How to capture |
|----|------|---------------------|----------------|
| **A1** | Nominal successful rollout (gym-aloha) | W&B run `auchm66k` (ACT nominal) — rollout media tab | Download a successful rollout MP4 from W&B. Alternatively, re-render with `roboeval evaluate --config configs/baseline/act_nominal.yaml` and capture the first success from `outputs/eval/act_nominal/videos/` |
| **A2** | +5 cm Recovery failure rollout | W&B run `w6k2wole` (ACT +5 cm) — rollout media tab | Same as above; find a clean Recovery failure |
| **A3** | Side-by-side composite of A1 + A2 | Build from A1 and A2 | `ffmpeg` `hstack`, see recipe below |
| **B1** | Spatial failure-mode stacked bar | `docs/figures/spatial_failure_distribution.png` | Already committed |
| **B2** | Cross-axis degradation curve | `docs/figures/cross_axis_degradation.png` | Already committed |
| **B3** | Phase 4 3-condition stacked bar | `docs/figures/phase4_ablation_failure_distribution.png` | Already committed |
| **C1** | Residual architecture diagram | The mermaid block in `docs/blog/2026-05-21-honest-null-residual.md` | Render to PNG via `npx @mermaid-js/mermaid-cli -i diagram.mmd -o C1.png` or screenshot from the GitHub-rendered blog post |
| **D1** | Live dashboard interaction (10 s) | https://rubenodechua-roboeval.hf.space/ | QuickTime → File → New Screen Recording → "Record Selected Portion" → drag a 1280×720 region over the browser → click through filters for 10 s |

---

## Shot capture walkthrough — step by step

This section expands every row of the shot list into a clickable procedure. Work
through it top to bottom; each shot writes one file into
`~/Desktop/roboeval-demo-assets/` so iMovie has a single drop-in folder later.

Before you start:

```bash
mkdir -p ~/Desktop/roboeval-demo-assets
cd ~/Desktop/roboeval-demo-assets
# Install ffmpeg once if you don't have it
brew install ffmpeg          # ~3 min on first install
ffmpeg -version | head -1    # sanity check
```

W&B entity for every run URL below: **`rdechua-university-of-san-francisco`**.

---

### A1 — Nominal successful rollout (target output: `A1_raw.mp4`, ~8 s)

**Source:** W&B run `auchm66k` (ACT, nominal cell).

1. Open <https://wandb.ai/rdechua-university-of-san-francisco/roboeval/runs/auchm66k>
   in a browser while signed in to W&B.
2. Click the **Media** tab (left sidebar; the panel icon with a filmstrip).
3. Find the panel titled `eval/videos` (or `rollouts/video`, whichever variant
   was logged for this run). Each tile is one rollout.
4. Use the **success filter:** sort or filter the table on the `success` metric
   = 1. If the table view is collapsed, click the gear icon → "Show success
   column" → sort descending. Pick any rollout where success=1.
5. Hover the chosen video tile → click the ⋮ menu (top-right corner of the
   tile) → **Download video**. It saves as something like
   `media_video_eval-videos_42_0_<hash>.mp4`.
6. Rename and move to the asset folder:
   ```bash
   mv ~/Downloads/media_video_eval-videos_*.mp4 ~/Desktop/roboeval-demo-assets/A1_raw.mp4
   ```
7. Trim to 8 s starting from a clean frame (skip the first ~1 s if the gym
   reset is jumpy):
   ```bash
   ffmpeg -ss 1 -t 8 -i A1_raw.mp4 -c copy A1.mp4
   ```
8. Sanity check it plays and is muted-ready (the rollout has no audio anyway):
   ```bash
   ffprobe A1.mp4 2>&1 | grep -E "Duration|Stream"
   ```
   Expect `Duration: 00:00:08.0x` and a single video stream, no audio.

**Fallback if W&B media is missing:** re-render locally.

```bash
cd /Users/rubenodehcua/Desktop/roboeval
roboeval evaluate --config configs/baseline/act_nominal.yaml --num-rollouts 5
# Outputs land under outputs/eval/act_nominal/videos/seed*/rollout_*.mp4
ls outputs/eval/act_nominal/videos/seed0/ | head
cp outputs/eval/act_nominal/videos/seed0/rollout_0_success.mp4 \
   ~/Desktop/roboeval-demo-assets/A1_raw.mp4
# Then run the same ffmpeg -ss 1 -t 8 trim from step 7 above.
```

---

### A2 — +5 cm Recovery failure rollout (target output: `A2_raw.mp4`, ~8 s)

**Source:** W&B run `w6k2wole` (ACT, +5 cm cell, Phase 4 condition A re-eval).

1. Open <https://wandb.ai/rdechua-university-of-san-francisco/roboeval/runs/w6k2wole>.
2. Click **Media** → find `eval/videos`.
3. Filter for **failures:** success=0. From those, you want a **Recovery**
   failure specifically — the gripper visibly passes _over_ the cube without
   closing on it. Skip Approach failures (gripper never gets near) and
   Grasp-and-drop failures (cube briefly lifts then falls).
   - The taxonomy auto-labels are in
     `data/taxonomy/auto_labels_w6k2wole.json` if it exists locally — you can
     `jq '.labels[] | select(.failure_mode=="Recovery") | .rollout_idx'`
     to get a list of rollout indices that are confirmed Recovery failures,
     then cross-reference the video panel by index.
4. Download the chosen tile (same ⋮ → Download video flow as A1).
5. Rename and trim:
   ```bash
   mv ~/Downloads/media_video_eval-videos_*.mp4 ~/Desktop/roboeval-demo-assets/A2_raw.mp4
   ffmpeg -ss 1 -t 8 -i A2_raw.mp4 -c copy A2.mp4
   ```
6. Visually verify: open `A2.mp4` in QuickTime, scrub to the moment the gripper
   closes on empty air past the cube. That's the money frame for Beat 1.

**Tip:** if the W&B run doesn't have rollout videos (this can happen on
re-evaluation runs where logging was turned off), re-render the cell:

```bash
roboeval evaluate --config configs/baseline/act_spatial_y+5cm.yaml --num-rollouts 50
# Filter for Recovery failures via the auto-classifier output:
roboeval taxonomy classify outputs/eval/act_spatial_y+5cm/eval_results_*.json \
  --out /tmp/auto_labels_a2.json
jq '.labels[] | select(.failure_mode=="Recovery") | .rollout_idx' /tmp/auto_labels_a2.json
# Pick one index and copy the matching video file
```

---

### A3 — Side-by-side composite (target output: `A3_sidebyside.mp4`)

Run the ffmpeg recipe in the next section (it operates on `A1.mp4` and
`A2.mp4` you just created). One command, ~5 s runtime:

```bash
cd ~/Desktop/roboeval-demo-assets
ffmpeg -i A1.mp4 -i A2.mp4 -filter_complex \
  "[0:v]scale=960:720,setsar=1[l];[1:v]scale=960:720,setsar=1[r];[l][r]hstack=inputs=2[out]" \
  -map "[out]" -an -c:v libx264 -crf 18 -pix_fmt yuv420p A3_sidebyside.mp4
```

Open `A3_sidebyside.mp4` in QuickTime to confirm the panels are aligned
(left = nominal, right = +5 cm). If one looks squashed, rerun with the source
order swapped or adjust the per-input `scale=` to match aspect.

---

### B1, B2, B3 — Already-committed PNGs (target outputs: copies in the asset folder)

The figures are already in the repo. Copy them into the asset folder so iMovie
sees them next to the videos:

```bash
cd /Users/rubenodehcua/Desktop/roboeval
cp docs/figures/spatial_failure_distribution.png      ~/Desktop/roboeval-demo-assets/B1.png
cp docs/figures/cross_axis_degradation.png            ~/Desktop/roboeval-demo-assets/B2.png
cp docs/figures/phase4_ablation_failure_distribution.png ~/Desktop/roboeval-demo-assets/B3.png

# Confirm they're 1080p-friendly (width ≥ 1280)
for f in ~/Desktop/roboeval-demo-assets/B?.png; do
  echo "$(basename "$f"): $(sips -g pixelWidth -g pixelHeight "$f" | tail -2 | xargs)"
done
```

If any PNG is narrower than 1280 px, upscale with `sips` or rerender at higher
DPI:

```bash
sips -Z 1920 ~/Desktop/roboeval-demo-assets/B1.png   # capped longest side at 1920
```

In iMovie, drop each PNG on the timeline and set the clip duration to the
beat's budget (B1 = 8 s, B2 = 7 s, B3 = 20 s).

---

### C1 — Residual architecture diagram (target output: `C1.png`)

**Option A — render the mermaid block to PNG (preferred; sharper).**

1. Save the exact mermaid block from the blog post to a file:
   ```bash
   cat > ~/Desktop/roboeval-demo-assets/diagram.mmd <<'EOF'
   flowchart LR
       O[observation o_t] --> ACT[Frozen ACT base]
       O --> R[Residual MLP δ_θ(o_t)]
       ACT --> S[a_base]
       R --> SCALE[× α = 0.05]
       S --> SUM((+))
       SCALE --> SUM
       SUM --> A[a_t = a_base + α · δ_θ]
   EOF
   ```
2. Render with the mermaid CLI (no global install needed; `npx` fetches on
   demand). The first run downloads ~80 MB of Chromium; subsequent runs are
   instant.
   ```bash
   cd ~/Desktop/roboeval-demo-assets
   npx -y @mermaid-js/mermaid-cli@latest -i diagram.mmd -o C1.png \
     --width 1920 --height 1080 --backgroundColor white
   ```
3. Open `C1.png` in Preview. Confirm the `α = 0.05` and `δ_θ(o_t)` labels are
   crisp (the script highlights them at 0:30).

**Option B — screenshot from the GitHub-rendered blog post (faster, lower DPI).**

1. Open
   <https://github.com/RDechua/roboeval/blob/main/docs/blog/2026-05-21-honest-null-residual.md>
   in Safari or Chrome.
2. Scroll to the **§5 Hypothesis** section — the mermaid diagram renders
   inline there.
3. Zoom the page to 200 % (`⌘ +` twice) so the screenshot will be ≥ 1080 px tall.
4. `⌘ + Shift + 4`, drag a rectangle around the diagram. macOS saves it to
   the Desktop as `Screen Shot YYYY-MM-DD at HH.MM.SS.png`.
5. Move and rename:
   ```bash
   mv "$HOME/Desktop/Screen Shot"*.png ~/Desktop/roboeval-demo-assets/C1.png
   ```

---

### D1 — Live dashboard interaction recording (target output: `D1.mov`, ~10 s)

1. Open the live dashboard in Chrome or Safari:
   <https://rubenodechua-roboeval.hf.space/>
   (Wait for the spinner to clear — first hit can cold-start the Space, ~5 s.)
2. Resize the browser window to roughly 1280×720. The fastest way: press
   `⌥ ⇧ ⌘ R` in Safari to reset zoom, then drag the window corner until the
   address-bar URL looks centred over a ~1280-px wide viewport. (You can also
   use a free utility like Rectangle to snap exact sizes.)
3. Open QuickTime Player → **File → New Screen Recording**. macOS opens the
   screenshot toolbar.
4. In the toolbar choose **Record Selected Portion** (third icon from the
   left). Drag a rectangle over the browser viewport — aim for the visible
   chart area, exclude the address bar and OS chrome.
5. Click **Options → Microphone: None** (you'll narrate separately in Step 1
   of the recording procedure). Click **Record**.
6. Perform this click-through in 10 seconds, smoothly:
   1. **t = 0 s:** the page is loaded showing the cross-axis degradation
      curve.
   2. **t = 2 s:** click the **Axis filter** dropdown → switch from "Both"
      to "Spatial". The right panel collapses.
   3. **t = 4 s:** click the **+5 cm** cell selector. The failure-mode stack
      below redraws.
   4. **t = 6 s:** click the **Recovery** legend swatch to toggle it off,
      then back on. Watch the stacked bars animate.
   5. **t = 9 s:** scroll down to reveal the Phase 4 ablation panel. Pause
      on a clear frame.
7. Press the stop icon in the menu bar (or `⌃ ⌘ Esc`). QuickTime opens the
   recording in a new window.
8. Trim with `⌘ T`: drag the yellow handles to keep just the 10 s of action,
   click **Trim**.
9. **File → Save** → name `D1.mov` → save to `~/Desktop/roboeval-demo-assets/`.
10. Optional: convert to `.mp4` so iMovie ingests faster:
    ```bash
    cd ~/Desktop/roboeval-demo-assets
    ffmpeg -i D1.mov -c:v libx264 -crf 20 -pix_fmt yuv420p -an D1.mp4
    ```

**If the cold-start spinner steals 3 s of your take:** hit reload, wait until
the first chart renders, _then_ start recording. The Space stays warm for
~5 min between hits.

---

### Final asset folder checklist

After all the steps above, `~/Desktop/roboeval-demo-assets/` should contain:

```
A1.mp4              # 8 s, nominal success
A2.mp4              # 8 s, +5 cm Recovery failure
A3_sidebyside.mp4   # 8 s, side-by-side composite
B1.png              # spatial failure-mode stacked bar
B2.png              # cross-axis degradation curve
B3.png              # Phase 4 3-condition stacked bar
C1.png              # residual architecture diagram
D1.mp4              # 10 s, live dashboard interaction
diagram.mmd         # mermaid source (optional, kept for re-render)
```

Drag the whole folder into iMovie's Project Media library and you're ready
for the **Recording procedure** section below.

---

## ffmpeg recipe — A3 side-by-side composite

If A1 and A2 have different lengths, trim to the shorter one first:

```bash
# Trim both to 8 seconds (match the Beat 1 budget)
ffmpeg -ss 0 -t 8 -i A1.mp4 -c copy A1_trim.mp4
ffmpeg -ss 0 -t 8 -i A2.mp4 -c copy A2_trim.mp4

# Stack side-by-side, scale each panel to 960x720 (fills 1920x720)
ffmpeg -i A1_trim.mp4 -i A2_trim.mp4 -filter_complex \
  "[0:v]scale=960:720,setsar=1[l];[1:v]scale=960:720,setsar=1[r];[l][r]hstack=inputs=2[out]" \
  -map "[out]" -an -c:v libx264 -crf 18 -pix_fmt yuv420p A3_sidebyside.mp4
```

Drop `A3_sidebyside.mp4` into iMovie as a single clip for Beat 1.

---

## Recording procedure (Mac, iMovie)

### One-time setup (~10 min)

1. Open **iMovie** → File → New Project → "Movie" template → name it `roboeval-demo`.
2. Project settings → set frame rate to **30 fps** and resolution to **1920×1080**.

### Step 1 — record the narration (~20 min)

1. QuickTime Player → File → New Audio Recording.
2. Select your microphone (built-in MacBook mic is fine; AirPods reduce reverb).
3. Read the five beats end-to-end without stopping. Re-record any line you flubbed.
4. Save as `narration.m4a`.
5. Drop `narration.m4a` into the iMovie project (Audio track 1).

### Step 2 — assemble the visuals (~60 min)

1. Drag the B-roll files (A3, B1, B2, B3, C1, D1) into the iMovie project library.
2. Place them on the video track aligned to the narration timestamps in this script.
3. For static images (B1, B2, B3, C1), set the clip duration to match the beat:
   - Click the image clip → use the inspector to set duration in seconds.
4. Enable the **Ken Burns** effect on B1 (slow zoom into the +5 cm column for Beat 2).
5. Add the on-screen text overlays:
   - Title browser → "Lower Third" style for the captions.
   - Manually type each on-screen label per the script.

### Step 3 — export (~5 min)

1. File → Share → File.
2. Format: Video and Audio.
3. Resolution: 1080p.
4. Quality: High.
5. Compress: Better Quality.
6. Save as `roboeval-demo-v1.mp4`.

### Step 4 — verify against PRD §9.1

- [ ] Total length ≤ 90 s (1:25 in the script leaves a 5 s safety buffer; final cut should land 1:20–1:30 max).
- [ ] Narration present throughout.
- [ ] Each failure mode that's discussed is annotated on-screen (text overlay or arrow).
- [ ] Output is 1920×1080 minimum (run `ffprobe roboeval-demo-v1.mp4` to confirm).
- [ ] No audio clipping (peaks below −3 dB).

---

## Hosting

**Recommended:** YouTube unlisted.

1. youtube.com/upload → upload `roboeval-demo-v1.mp4`.
2. Visibility: **Unlisted** (anyone with the link can view; not in search).
3. Title: `RoboEval — 90s demo · honest null on ACT residual RL`.
4. Description: paste the blog post lede + link to the dashboard.
5. Save the URL.

Then add a video badge to the top-level `README.md`:

```markdown
[![Demo](https://img.shields.io/badge/Video-90s%20demo-red?logo=youtube)](https://youtu.be/<id>)
```

And add the embed to the dashboard `analysis/dashboard/README.md` under "First visit".

**Alternative if you don't want a YouTube account:** commit the MP4 to git via Git LFS. ~30 MB for a 1:25 1080p clip at iMovie's "High" preset; well within GitHub's free LFS quota.

---

## When you're done

Tell me the YouTube URL (or LFS commit SHA) and I'll:

1. Add the badge to top-level `README.md`.
2. Embed the video in the dashboard `README.md` (HF Space supports HTML `<iframe>` embeds in markdown).
3. Update `docs/STATE.md` to mark Phase 5 deliverable #2 closed.

---

## Total estimated effort

- Recording narration: 20 min
- Pulling W&B rollouts + ffmpeg composite: 30 min
- iMovie assembly: 60 min
- Export + upload: 10 min
- **Total: ~2 hr.** Front-load the narration recording — getting that clean is the slowest variable.
