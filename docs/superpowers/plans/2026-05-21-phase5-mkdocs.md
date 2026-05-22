# Phase 5 MkDocs Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a static `mkdocs-material` site at `https://rdechua.github.io/roboeval/` with a 4-tab structure (Home / Project / API Reference / Blog), auto-deployed via GitHub Action on push to `main`.

**Architecture:** Single `mkdocs.yml` at repo root, lean nav, mkdocstrings-powered API reference (one stub per top-level submodule). `mkdocs build --strict` gates every commit. Deploy via dedicated `docs.yml` workflow that runs after the existing `ci.yml`; existing CI stays untouched.

**Tech Stack:** mkdocs 1.6+, mkdocs-material 9.5+, mkdocstrings[python] 0.25+. All three already declared in `pyproject.toml`'s `docs` optional extra.

**Workflow gates per commit:** `ruff check`, `ruff format --check`, `mypy --strict roboeval`, `pytest -q`. The new docs workflow adds `mkdocs build --strict` as a deploy-time gate. Commit author `Rubeno Dechua <rubenodechua123@gmail.com>`, no Claude trailers.

---

## File Structure

| Path | Purpose |
|---|---|
| `mkdocs.yml` | Single config file: theme, plugins, nav, markdown extensions |
| `docs/index.md` | Landing page (new) |
| `docs/getting-started.md` | Quickstart promoted from README (new) |
| `docs/reference/index.md` | API reference overview (new) |
| `docs/reference/cli.md` | mkdocstrings stub for `roboeval.cli` (new) |
| `docs/reference/envs.md` | mkdocstrings stub for `roboeval.envs` (new) |
| `docs/reference/evaluation.md` | mkdocstrings stub for `roboeval.evaluation` (new) |
| `docs/reference/policies.md` | mkdocstrings stub for `roboeval.policies` (new) |
| `docs/reference/taxonomy.md` | mkdocstrings stub for `roboeval.taxonomy` (new) |
| `docs/reference/residual.md` | mkdocstrings stub for `roboeval.residual` (new) |
| `docs/reference/dashboard.md` | mkdocstrings stub for `roboeval.dashboard` (new) |
| `tests/docs/__init__.py` | Test package (new) |
| `tests/docs/test_mkdocs_build.py` | `mkdocs build --strict` smoke test (new) |
| `.github/workflows/docs.yml` | Build + deploy workflow (new) |
| `README.md` | Add "Docs" badge |
| `docs/STATE.md` | Close MkDocs deliverable item |

---

## Task 1: Install docs extras locally and verify the toolchain

**Files:** none modified — verification only.

- [ ] **Step 1: Install the docs extra into the project venv**

```bash
/Users/rubenodehcua/.local/bin/uv pip install \
  --python /Users/rubenodehcua/Desktop/roboeval/.venv/bin/python \
  -e '.[docs]'
```

Expected: `mkdocs`, `mkdocs-material`, `mkdocstrings[python]` installed; output ends with "Installed N packages".

- [ ] **Step 2: Verify the binaries are callable**

```bash
.venv/bin/python -m mkdocs --version
.venv/bin/python -c "import mkdocs_material, mkdocstrings; print('ok')"
```

Expected: a version string from mkdocs (e.g., `mkdocs, version 1.6.1`) and `ok` from the import check.

No commit — local-only setup.

---

## Task 2: Write `mkdocs.yml`

**Files:**
- Create: `mkdocs.yml`

- [ ] **Step 1: Write the configuration**

```yaml
# mkdocs.yml
site_name: RoboEval
site_description: Failure-mode study and residual RL for ACT/Diffusion Policy on AlohaTransferCube.
site_url: https://rdechua.github.io/roboeval/
repo_url: https://github.com/RDechua/roboeval
repo_name: RDechua/roboeval
edit_uri: edit/main/docs/

theme:
  name: material
  features:
    - navigation.tabs
    - navigation.sections
    - navigation.expand
    - navigation.top
    - search.highlight
    - search.suggest
    - content.code.copy
    - content.action.edit
  palette:
    - media: "(prefers-color-scheme: light)"
      scheme: default
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-7
        name: Switch to dark mode
    - media: "(prefers-color-scheme: dark)"
      scheme: slate
      primary: indigo
      accent: indigo
      toggle:
        icon: material/brightness-4
        name: Switch to light mode
  icon:
    repo: fontawesome/brands/github

plugins:
  - search
  - mkdocstrings:
      handlers:
        python:
          paths: [.]
          options:
            show_source: true
            show_root_heading: true
            members_order: source
            separate_signature: true
            docstring_style: google
            show_signature_annotations: true
            heading_level: 2

markdown_extensions:
  - admonition
  - pymdownx.details
  - pymdownx.superfences:
      custom_fences:
        - name: mermaid
          class: mermaid
          format: !!python/name:pymdownx.superfences.fence_code_format
  - pymdownx.tabbed:
      alternate_style: true
  - tables
  - toc:
      permalink: true

nav:
  - Home: index.md
  - Project:
      - PRD: PRD.md
      - Getting started: getting-started.md
  - API Reference:
      - Overview: reference/index.md
      - CLI: reference/cli.md
      - envs: reference/envs.md
      - evaluation: reference/evaluation.md
      - policies: reference/policies.md
      - taxonomy: reference/taxonomy.md
      - residual: reference/residual.md
      - dashboard: reference/dashboard.md
  - Blog:
      - "Honest null on +5cm": blog/2026-05-21-honest-null-residual.md

not_in_nav: |
  STATE.md
  phase4_ablation.md
  research-log.md
  superpowers/**
  figures/**

extra:
  social:
    - icon: fontawesome/brands/github
      link: https://github.com/RDechua/roboeval
    - icon: fontawesome/brands/hugging-face
      link: https://huggingface.co/spaces/RubenoDechua/roboeval
```

- [ ] **Step 2: Smoke build (expected to fail on missing nav targets)**

```bash
.venv/bin/python -m mkdocs build --strict --site-dir /tmp/mkdocs-smoke 2>&1 | tail -20
```

Expected: failure with messages about missing `index.md`, `getting-started.md`, `reference/*.md`. That's the point — the nav block references files that don't exist yet. The next tasks create them.

- [ ] **Step 3: Commit just the config**

```bash
git add mkdocs.yml
git commit -m "feat(docs): mkdocs.yml with material theme + 4-tab nav"
```

(No push yet — the next task makes the build pass before remote sees it.)

---

## Task 3: Write the landing page (`docs/index.md`)

**Files:**
- Create: `docs/index.md`

- [ ] **Step 1: Write the landing page**

```markdown
# RoboEval

Failure-mode study and residual RL for open-source robot-learning policies on the bimanual ALOHA Transfer Cube task.

[![Live Demo](https://img.shields.io/badge/HF%20Spaces-Live%20Demo-blue?logo=huggingface)](https://huggingface.co/spaces/RubenoDechua/roboeval)
[![Writeup](https://img.shields.io/badge/Blog-Honest%20null-blueviolet)](blog/2026-05-21-honest-null-residual.md)
[![Repo](https://img.shields.io/badge/GitHub-Repo-181717?logo=github)](https://github.com/RDechua/roboeval)

## What this is

RoboEval is a typed Python codebase plus a public-facing research artifact. It measures where ACT
breaks under realistic perturbation (spatial cube shifts, action delays), classifies failure modes
into six operational categories, and ships a residual RL ablation that attempts to recover the
top-frequency failure — with an honest null result and a documented v1.1 fix path.

## Where to go

- **[Live dashboard](https://huggingface.co/spaces/RubenoDechua/roboeval)** — interactive degradation curves + Phase 4 ablation.
- **[Blog post](blog/2026-05-21-honest-null-residual.md)** — the honest-null writeup (~2000 words).
- **[Getting started](getting-started.md)** — `git clone` to first rollout in five commands.
- **[API Reference](reference/index.md)** — auto-generated from the typed `roboeval/` package.
- **[Product spec](PRD.md)** — the requirements doc the whole project ships against.

## Stack

Python 3.11 · LeRobot · MuJoCo · Stable-Baselines3 · Hydra · Weights & Biases · Plotly/Dash · MkDocs.
M1 MPS for inference and PPO training — no CUDA required.

## License

MIT.
```

- [ ] **Step 2: Verify the landing page contributes to a passing build (still need the other pages)**

```bash
.venv/bin/python -m mkdocs build --strict --site-dir /tmp/mkdocs-smoke 2>&1 | grep -E "error|warning" | head -5
```

Expected: the index.md missing-target error is gone; other missing-target errors remain.

- [ ] **Step 3: Commit**

```bash
git add docs/index.md
git commit -m "feat(docs): landing page"
```

---

## Task 4: Write the Getting Started page (`docs/getting-started.md`)

**Files:**
- Create: `docs/getting-started.md`

- [ ] **Step 1: Write the quickstart**

```markdown
# Getting started

Five commands to a working rollout on a fresh clone (Python 3.11, macOS M1 or Linux x86).

## 1. Clone

```bash
git clone https://github.com/RDechua/roboeval.git
cd roboeval
```

## 2. Create the venv and install

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

(If you don't have `uv`, `python3.11 -m venv .venv && pip install -e '.[dev]'` works too — `uv` is just faster.)

## 3. Verify the stack

```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available())"
python -c "from lerobot.policies.act.modeling_act import ACTPolicy; print('ACT loaded')"
```

Expected: `mps: True` on M1 (or `False` on Linux — both are fine, `torch` falls back to CPU). `ACT loaded` confirms the LeRobot 0.4.x namespace works.

## 4. First smoke rollout

```bash
roboeval smoke --steps 10
```

This runs 10 random-action steps against `gym_aloha/AlohaTransferCube-v0` and prints a per-step trace. If it completes without raising, the dependency stack is healthy.

## 5. Real evaluation (slower — about 50 minutes on M1)

```bash
roboeval evaluate --config configs/baseline/act_nominal.yaml
```

This loads ACT, runs 3 seeds × 50 rollouts on the nominal cell, classifies the rollouts via the
PRD §7.2 taxonomy, and writes a schema-v1 `eval_results_<run_id>.json` plus an `auto_labels_<run_id>.json`
to `outputs/eval/act_nominal/` and `data/taxonomy/` respectively.

## What's next

- **Train a residual** with `roboeval residual train --config configs/residual/residual_ppo_y+5cm_sparse.yaml`.
- **Launch the dashboard locally** with `roboeval dashboard run` — opens http://localhost:8050.
- **Read the writeup** at [blog/2026-05-21-honest-null-residual.md](blog/2026-05-21-honest-null-residual.md).

## Quality gates

The repo's CI runs four gates on every push to `main` and every PR:

```bash
ruff check .
ruff format --check .
mypy --strict roboeval
pytest -q
```

Each should exit 0. If any complains on your machine, open an issue — the gates pass cleanly on
the maintained branches.
```

- [ ] **Step 2: Verify**

```bash
.venv/bin/python -m mkdocs build --strict --site-dir /tmp/mkdocs-smoke 2>&1 | grep -E "error|warning" | head -5
```

Expected: getting-started.md missing-target error is gone.

- [ ] **Step 3: Commit**

```bash
git add docs/getting-started.md
git commit -m "feat(docs): getting-started quickstart"
```

---

## Task 5: Write the API Reference overview (`docs/reference/index.md`)

**Files:**
- Create: `docs/reference/index.md`

- [ ] **Step 1: Create the directory and the overview page**

```bash
mkdir -p docs/reference
```

```markdown
# API Reference

Auto-generated from the typed Python source in [`roboeval/`](https://github.com/RDechua/roboeval/tree/main/roboeval).
All modules pass `mypy --strict` and follow the Google docstring style.

| Module | What's in here |
|---|---|
| **[CLI](cli.md)** | `roboeval smoke`, `evaluate`, `calibrate`, `residual {train,evaluate,aggregate}`, `dashboard {build,run}` — every command the user touches. |
| **[envs](envs.md)** | gym-aloha env factory, success criterion (geometric + dwell), spatial / temporal perturbation wrappers. |
| **[evaluation](evaluation.md)** | rollout engine, multi-seed loop, calibration, config loader, W&B logger, schema-v1 `eval_results_<run_id>.json` writer. |
| **[policies](policies.md)** | `Policy` protocol, ACT loader, policy factory. |
| **[taxonomy](taxonomy.md)** | six-category failure-mode classifier, Cohen's κ agreement, schema-v1 `auto_labels_<run_id>.json` writer. |
| **[residual](residual.md)** | MLP residual + compositor, reward functions, env wrapper, SB3 PPO training loop, Phase 4 aggregator (Welch's t + bootstrap CI). |
| **[dashboard](dashboard.md)** | Pure dashboard logic: dataclasses (cells, ablation), JSON loaders, Plotly figure builders. |

## Reading these pages

Each submodule page is a single mkdocstrings directive that renders every public function and
class with its full signature, type annotations, and Google-style docstring. The "source" link
on each symbol opens the canonical Python file on GitHub. Use the in-page table of contents to
jump to a specific function.
```

- [ ] **Step 2: Verify**

```bash
.venv/bin/python -m mkdocs build --strict --site-dir /tmp/mkdocs-smoke 2>&1 | grep -E "error|warning" | head -5
```

Expected: reference/index.md missing-target error gone; the 7 submodule pages still missing.

- [ ] **Step 3: Commit**

```bash
git add docs/reference/index.md
git commit -m "feat(docs): API reference overview"
```

---

## Task 6: Create the 7 mkdocstrings submodule stubs

**Files:**
- Create: `docs/reference/cli.md`
- Create: `docs/reference/envs.md`
- Create: `docs/reference/evaluation.md`
- Create: `docs/reference/policies.md`
- Create: `docs/reference/taxonomy.md`
- Create: `docs/reference/residual.md`
- Create: `docs/reference/dashboard.md`

- [ ] **Step 1: Write `docs/reference/cli.md`**

````markdown
# CLI

Command-line entry points. Run `roboeval <subcommand> --help` for full argument lists.

::: roboeval.cli
````

- [ ] **Step 2: Write `docs/reference/envs.md`**

````markdown
# Environments

gym-aloha env factory, the geometric success criterion, and perturbation wrappers (spatial cube
translation, temporal action-chunk delay).

::: roboeval.envs
````

- [ ] **Step 3: Write `docs/reference/evaluation.md`**

````markdown
# Evaluation

Rollout engine, multi-seed loop, Hydra config loader, W&B logger, calibration, schema-v1 results
I/O. The CLI's `evaluate` and `residual evaluate` subcommands compose these primitives.

::: roboeval.evaluation
````

- [ ] **Step 4: Write `docs/reference/policies.md`**

````markdown
# Policies

The `Policy` protocol (declares `policy_id` + `device` + `select_action`) plus a thin LeRobot ACT
adapter. The policy factory swaps implementations behind a single `kind:` config flag.

::: roboeval.policies
````

- [ ] **Step 5: Write `docs/reference/taxonomy.md`**

````markdown
# Failure-mode taxonomy

The six PRD §7.2 failure categories (Success, Grasp, Approach, Recovery, Oscillation, Timeout,
Visual confusion, Needs review), the rule-based classifier, Cohen's κ inter-rater agreement, and
the schema-v1 `auto_labels_<run_id>.json` writer.

::: roboeval.taxonomy
````

- [ ] **Step 6: Write `docs/reference/residual.md`**

````markdown
# Residual RL

`ResidualMLP` (2×256 GELU) + `ResidualCompositor`, sparse / shaped / combined rewards, the gym
wrapper that composes the frozen base with the residual, the SB3 PPO training loop, and the Phase
4 ablation aggregator (Welch's t + bootstrap CI, stdlib-only).

::: roboeval.residual
````

- [ ] **Step 7: Write `docs/reference/dashboard.md`**

````markdown
# Dashboard

Pure data + figure logic for the Phase 5 Plotly/Dash app. The Dash skeleton itself lives at
`analysis/dashboard/app.py`; this package contains only the loaders, dataclasses, and figure
builders so it can be unit-tested under `mypy --strict` without importing `dash`.

::: roboeval.dashboard
````

- [ ] **Step 8: Strict-build the whole site**

```bash
.venv/bin/python -m mkdocs build --strict --site-dir /tmp/mkdocs-smoke 2>&1 | tail -10
```

Expected: build succeeds. If any warning fires (broken link, deprecated option, missing
docstring on a public symbol), fix it before committing. Common fixes:

- mkdocstrings can't find a module → check `mkdocs.yml`'s `plugins.mkdocstrings.handlers.python.paths` includes `.` (it does).
- A doctring uses unknown ReST syntax → confirm `docstring_style: google` matches the codebase.

- [ ] **Step 9: Spot-check the rendered site**

```bash
.venv/bin/python -m mkdocs serve --dev-addr 127.0.0.1:8765 &
sleep 3
curl -sI http://127.0.0.1:8765/ | head -1
curl -s http://127.0.0.1:8765/reference/cli/ | grep -oE '<h2[^>]*>[^<]+</h2>' | head -5
kill %1 2>/dev/null
```

Expected: `HTTP/1.0 200 OK` and several `<h2>` tags listing CLI functions.

- [ ] **Step 10: Commit**

```bash
git add docs/reference/cli.md docs/reference/envs.md docs/reference/evaluation.md \
        docs/reference/policies.md docs/reference/taxonomy.md docs/reference/residual.md \
        docs/reference/dashboard.md
git commit -m "feat(docs): mkdocstrings stubs for 7 submodules"
```

---

## Task 7: Add the `mkdocs build --strict` test

**Files:**
- Create: `tests/docs/__init__.py`
- Create: `tests/docs/test_mkdocs_build.py`

- [ ] **Step 1: Create the test package**

```python
# tests/docs/__init__.py
```

(Empty file.)

- [ ] **Step 2: Write the build test**

```python
# tests/docs/test_mkdocs_build.py
"""Smoke test: `mkdocs build --strict` succeeds.

Skipped automatically on environments without `mkdocs` installed (e.g., the
minimal CI image used by ``.github/workflows/ci.yml``). The dedicated
``docs.yml`` workflow installs ``.[docs]`` so this test runs there.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs")


def test_mkdocs_build_strict_succeeds(tmp_path: Path) -> None:
    """`mkdocs build --strict` exits 0 against the committed config."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mkdocs",
            "build",
            "--strict",
            "--site-dir",
            str(tmp_path / "site"),
            "--config-file",
            str(repo_root / "mkdocs.yml"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"mkdocs build --strict failed:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    # The built index must exist.
    assert (tmp_path / "site" / "index.html").exists()
```

- [ ] **Step 3: Run the test**

```bash
.venv/bin/pytest tests/docs/test_mkdocs_build.py -v
```

Expected: 1 passed.

- [ ] **Step 4: Run all four gates**

```bash
.venv/bin/ruff check tests/docs
.venv/bin/ruff format --check tests/docs
.venv/bin/mypy --strict roboeval
.venv/bin/pytest -q
```

Expected: all green. The new test is gated by `pytest.importorskip` so the existing ci.yml (which doesn't install `.[docs]`) skips it cleanly; the new docs.yml workflow installs `.[docs]` so the test runs there.

- [ ] **Step 5: Commit**

```bash
git add tests/docs/__init__.py tests/docs/test_mkdocs_build.py
git commit -m "test(docs): mkdocs build --strict smoke test"
```

---

## Task 8: Add the docs deploy workflow

**Files:**
- Create: `.github/workflows/docs.yml`

- [ ] **Step 1: Write the workflow**

```yaml
# .github/workflows/docs.yml
name: Docs

on:
  push:
    branches: [main]
    paths:
      - "docs/**"
      - "roboeval/**"
      - "mkdocs.yml"
      - "pyproject.toml"
      - ".github/workflows/docs.yml"
  workflow_dispatch:

permissions:
  contents: write

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # gh-deploy needs the full history to push to gh-pages

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install docs extras
        run: |
          python -m pip install --upgrade pip
          # The mkdocstrings handler imports the package it documents, so
          # install the project itself plus the docs extra.
          pip install --index-url https://download.pytorch.org/whl/cpu \
                      "torch>=2.0"
          pip install -e '.[docs]'

      - name: Build (strict)
        run: mkdocs build --strict

      - name: Deploy to gh-pages
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          mkdocs gh-deploy --force --remote-branch gh-pages
```

- [ ] **Step 2: Sanity-check locally that the strict build still works**

```bash
.venv/bin/python -m mkdocs build --strict --site-dir /tmp/mkdocs-smoke 2>&1 | tail -3
```

Expected: build succeeds.

- [ ] **Step 3: Commit and push**

```bash
git add .github/workflows/docs.yml
git commit -m "ci(docs): mkdocs build + gh-deploy workflow"
git push origin main
```

This is the first push of the MkDocs work. CI will run both `ci.yml` (existing, unchanged) and the new `docs.yml`. Watch both at https://github.com/RDechua/roboeval/actions.

- [ ] **Step 4: Verify the deploy ran**

After the workflow finishes, check that the `gh-pages` branch exists:

```bash
git fetch origin gh-pages 2>&1 | tail -3
git log --oneline -1 origin/gh-pages
```

Expected: the branch exists with one initial commit.

---

## Task 9: Enable GitHub Pages (manual, one-time)

**Files:** none — this is a repo settings change in the GitHub UI.

- [ ] **Step 1: Open repo settings**

Navigate to https://github.com/RDechua/roboeval/settings/pages.

- [ ] **Step 2: Configure Pages**

In "Build and deployment":
- Source: **Deploy from a branch**
- Branch: **`gh-pages`** / **`/ (root)`** → Save.

Pages will publish at `https://rdechua.github.io/roboeval/`. The first publish takes 1–3 minutes; subsequent pushes that touch docs paths re-deploy in under a minute.

- [ ] **Step 3: Verify the site is live**

```bash
curl -sI https://rdechua.github.io/roboeval/ | head -3
```

Expected: `HTTP/2 200`. If you see `HTTP/2 404`, wait another minute and retry — the first deploy can take up to 3 minutes for DNS / CDN warming.

No commit — this is a settings change, not a code change.

---

## Task 10: README badge + STATE.md closure

**Files:**
- Modify: `README.md`
- Modify: `docs/STATE.md`

- [ ] **Step 1: Add the Docs badge to top-level `README.md`**

Open `README.md` and insert the Docs badge after the existing Live Demo + Writeup badges:

```markdown
[![Live Demo](https://img.shields.io/badge/HF%20Spaces-Live%20Demo-blue?logo=huggingface)](https://huggingface.co/spaces/RubenoDechua/roboeval)
[![Writeup](https://img.shields.io/badge/Blog-Honest%20null-blueviolet)](docs/blog/2026-05-21-honest-null-residual.md)
[![Docs](https://img.shields.io/badge/Docs-mkdocs--material-526CFE?logo=materialformkdocs)](https://rdechua.github.io/roboeval/)
```

- [ ] **Step 2: Update `docs/STATE.md`** — close the MkDocs item in "Next session intent"

Find this block:

```markdown
4. **MkDocs site** — static-site wrapper around PRD, research-log,
   phase4_ablation.md, plus auto-generated API docs.
```

Replace with:

```markdown
4. **MkDocs site landed** ✓ — live at https://rdechua.github.io/roboeval/.
   Lean 4-tab nav (Home / Project / API / Blog), 7 mkdocstrings submodule
   pages, auto-deploy via `.github/workflows/docs.yml` on push to main.
   `mkdocs build --strict` enforced as a CI gate.
```

- [ ] **Step 3: Run all four gates**

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy --strict roboeval
.venv/bin/pytest -q
```

Expected: all green.

- [ ] **Step 4: Commit and push**

```bash
git add README.md docs/STATE.md
git commit -m "docs(phase5): README docs badge + STATE.md closure"
git push origin main
```

The push triggers `docs.yml` again — verify the re-deploy publishes the README badge change to the gh-pages branch in <60 s.

- [ ] **Step 5: Final acceptance check (manual)**

Open https://rdechua.github.io/roboeval/ in a browser and verify:

- [ ] 4 top-level tabs visible: Home / Project / API Reference / Blog.
- [ ] Dark-mode toggle works.
- [ ] `/reference/cli/` renders at least one function with signature + docstring.
- [ ] `/blog/2026-05-21-honest-null-residual/` renders all 3 figures (the mermaid diagram included).
- [ ] `/getting-started/` shows the 5 numbered steps with copy buttons on code blocks.
- [ ] Edit pencil icon next to a page title opens the file on github.com.

If anything fails, fix in a follow-up commit rather than amending.

---

## Self-Review

**Spec coverage (against `2026-05-21-phase5-mkdocs-design.md`):**

| Spec section | Plan task(s) |
|---|---|
| §1 Goal | Tasks 1–10 collectively |
| §2 Audience (engineering due-diligence) | Task 3 (landing tone) + Task 5 (API overview voice) |
| §3 Scope (4 sections + `not_in_nav`) | Task 2 (`nav:` + `not_in_nav:` blocks); Tasks 3–6 create the 4 sections' pages |
| §4 Tech stack (mkdocs / material / mkdocstrings) | Task 1 install + Task 2 plugin config |
| §5 `mkdocs.yml` config | Task 2 |
| §6 New pages (10 total) | Task 3 (index), Task 4 (getting-started), Task 5 (reference/index), Task 6 (7 submodule stubs) |
| §7 CI / deploy | Task 8 (`docs.yml` workflow) + Task 9 (Pages settings) |
| §8 Testing (`mkdocs build --strict` smoke) | Task 7 |
| §9 Edge cases (mermaid fence, relative images) | Task 2 (`pymdownx.superfences` mermaid block) |
| §10 Acceptance | Task 10 Step 5 (manual checklist) |
| §11 Out of scope | Honored — no tasks for custom domain, versioned docs, custom theming |

**Placeholder scan:** none. Every step contains the actual content or command to commit.

**Type consistency:** the mkdocstrings module references (`roboeval.cli`, `roboeval.envs`, `roboeval.evaluation`, `roboeval.policies`, `roboeval.taxonomy`, `roboeval.residual`, `roboeval.dashboard`) match the package layout in STATE.md and the spec §3. No drift.

**Risks noted, not blocking:**

- Task 8's `pip install -e '.[docs]'` step also installs `torch>=2.0` because the package depends on it transitively; that's a ~200 MB CPU wheel download per workflow run (cached by pip after the first). Acceptable for a docs build.
- The first `mkdocs gh-deploy` in Task 8 will fail without write permissions; the `permissions: contents: write` block in the workflow grants it. If the workflow still fails on the first run with a 403, double-check Settings → Actions → General → Workflow permissions is set to "Read and write permissions."
- Task 9 (Pages settings) is a manual step. If the user prefers to skip it temporarily, the workflow will still publish to the `gh-pages` branch — the site just won't be reachable at the public URL until Pages is enabled.
