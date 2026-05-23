# Demo Video — Script, Shot List, and Recording Recipe

**Target:** 90-second narrated video, 1920×1080 minimum, MP4 (H.264).
**Acceptance (PRD §9.1):** narrated · failure modes visually annotated · ≤ 90 s · 1080p min.

Five beats, ~225 words at 2.5 words/sec. Time-stamps below assume the cuts hit ±0.5 s.

---

## Beat 1 · Hook (0:00 → 0:10, ~25 words)

**Narration:**

> ACT is a state-of-the-art imitation policy on the bimanual ALOHA Transfer Cube. At nominal,
> it succeeds 80% of the time. Shift the cube five centimeters, and 59% of the failures are
> the gripper sweeping right past it.

**Visual:** Full-screen B1 with a slow Ken Burns zoom.
- **0:00 → 0:04:** B1 — `docs/figures/spatial_failure_distribution.png`. Start at full-chart framing so the audience reads the axis (cells −5 → +5 cm).
- **0:04 → 0:10:** Zoom slowly into the +5 cm column. The Recovery slice (dominant colour) fills the frame by 0:09.
- **On-screen text** at 0:06 (large, centred, bold over translucent backdrop): "+5 cm spatial shift → 59% Recovery failures".

**Cut at 0:10** continues into Beat 2 on the same chart (no scene change; the camera just pulls back out).

*Note:* the original Beat 1 design called for side-by-side rollout video (A1 nominal vs. A2 +5 cm failure). The eval harness doesn't render or log video, so no rollout footage exists. B1 alone carries the hook — the +5 cm Recovery column is the visual equivalent of "robot misses."

---

## Beat 2 · The taxonomy (0:10 → 0:25, ~35 words)

**Narration:**

> I classified 150 rollouts per cell into six failure categories. Recovery — the gripper passing
> over the cube without engaging — dominates. It accounts for 59% of failures at +5 cm and
> emerges on both spatial and temporal perturbation axes.

**Visual:**
- **0:10 → 0:18:** Pull the Ken Burns zoom back out to full-chart B1, then pan slowly across all seven cells (−5 cm → +5 cm) so the audience sees the failure-mode distribution shift across the perturbation axis.
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
| **B1** | Spatial failure-mode stacked bar | `docs/figures/spatial_failure_distribution.png` | Already committed |
| **B2** | Cross-axis degradation curve | `docs/figures/cross_axis_degradation.png` | Already committed |
| **B3** | Phase 4 3-condition stacked bar | `docs/figures/phase4_ablation_failure_distribution.png` | Already committed |
| **C1** | Residual architecture diagram | The mermaid block in `docs/blog/2026-05-21-honest-null-residual.md` | Render to PNG via `npx @mermaid-js/mermaid-cli -i diagram.mmd -o C1.png` or screenshot from the GitHub-rendered blog post |
| **D1** | Live dashboard interaction (10 s) | https://rubenodechua-roboeval.hf.space/ | QuickTime → File → New Screen Recording → "Record Selected Portion" → drag a 1280×720 region over the browser → click through filters for 10 s |

> **A1, A2, A3 were dropped.** The eval harness never logged or rendered rollout video, so there's no nominal-success or +5 cm-failure clip to pull. Beat 1 now opens on B1 (zoomed +5 cm column) instead of a side-by-side picture-in-picture. If you ever want the rollout footage back, add a `--record-video` flag to `roboeval evaluate` using mujoco's offscreen renderer, then re-run the nominal and +5 cm evals.

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
       O["observation o_t"] --> ACT["Frozen ACT base"]
       O --> R["Residual MLP δ_θ(o_t)"]
       ACT --> S["a_base"]
       R --> SCALE["× α = 0.05"]
       S --> SUM((+))
       SCALE --> SUM
       SUM --> A["a_t = a_base + α · δ_θ"]
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
6. Perform this 7-step click-through in 10 seconds. The sequence is
   chosen for maximum visible state change per click (the cell dropdown
   is the money click — it reshapes the failure-mode bar dramatically):
   1. **t = 0.0 s:** page loaded on the Degradation curves panel ("Both"
      axis filter, "TSR (custom)" metric — both defaults). Cursor visible
      near the top.
   2. **t = 2.0 s:** move cursor down to the **cell dropdown** (currently
      "y+5cm"); click to open the menu.
   3. **t = 3.0 s:** select **"nominal"**. The big stacked bar reshapes
      from ~75% blue (Recovery) to ~80% green (Success). This is the
      money frame — hold the cursor still here.
   4. **t = 5.0 s:** click the dropdown again; select **"y+5cm"**. Bar
      snaps back to blue-dominated. Viewer now reads: each cell tells a
      different story.
   5. **t = 6.0 s:** smooth scroll down to reveal the **Phase 4 ablation**
      panel (three bars: A=base, B=sparse, C=shaped).
   6. **t = 8.0 s:** hover the cursor over the **orange slice in
      condition B** (the Approach growth). Plotly tooltip pops with the
      count.
   7. **t = 9.0 → 10.0 s:** hold the cursor still on the tooltip; freeze
      frame. End on the three ablation bars + the popped tooltip.

   **Things to skip on camera:** the "TSR (custom) / TSR (env) / TTS"
   metric radios (changes are too subtle in <2 s), the legend swatch
   toggles (tiny click target, looks fiddly), and the "Methods &
   reproducibility" disclosure (11 lines of run IDs, unreadable in 2 s).

   **Cold-start warm-up:** open the dashboard 5 min before recording so
   HF Spaces wakes up. Reload right before pressing Record so the
   selectors reset to default state ("Both" + "y+5cm"). The Space stays
   alive ~5 min idle.
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

1. Drag the B-roll files (B1, B2, B3, C1, D1) into the iMovie project library.
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
- Rendering C1 (mermaid → PNG) + recording D1 (dashboard click-through): 15 min
- iMovie assembly: 60 min
- Export + upload: 10 min
- **Total: ~1 hr 45 min.** Front-load the narration recording — getting that clean is the slowest variable.
