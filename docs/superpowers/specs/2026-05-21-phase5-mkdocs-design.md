# Phase 5 — MkDocs Documentation Site (Design)

**Status:** approved 2026-05-21 · **Owner:** Rubeno Dechua · **Targets:** PRD §9 (MkDocs Site deliverable), PRD §9.1 (quality checklist: all public functions documented; quickstart runs without modification on a fresh clone), PRD §4.1 (audience: engineering due-diligence).

## 1. Goal

Ship a static `mkdocs-material` site at `https://rdechua.github.io/roboeval/` that documents the project end-to-end: a landing page, the PRD, a runnable quickstart, auto-generated API docs for every `roboeval/` submodule, and the Phase 5 blog post. The site is the engineering-due-diligence-grade artifact for the project — distinct from the recruiter-grade live dashboard and the writeup-grade blog post.

PRD §9.1 acceptance:

- All public functions documented.
- Quickstart example runs without modification on a fresh clone.

## 2. Audience

Primary: engineering due-diligence — interviewers, code reviewers, and downstream library consumers who want to read API docs and reproduce a build. Secondary: future-me and contributors.

## 3. Scope (page set)

**Lean.** Four top-level sections:

| Section | Source | Notes |
|---|---|---|
| **Home** | `docs/index.md` (new) | One-paragraph pitch + 3 hero links (Live Demo, Blog, API). |
| **Project / PRD** | `docs/PRD.md` (existing) | Render as-is. |
| **Project / Getting started** | `docs/getting-started.md` (new) | Quickstart promoted from README footer to a first-class doc. |
| **API Reference** | `docs/reference/{index,cli,envs,evaluation,policies,taxonomy,residual,dashboard}.md` (new) | One stub per top-level submodule + an overview index. |
| **Blog** | `docs/blog/2026-05-21-honest-null-residual.md` (existing) | Render as-is. |

**Out of the nav (hidden, but still rendered when linked):** `docs/STATE.md`, `docs/phase4_ablation.md`, `docs/research-log.md`, anything under `docs/superpowers/**`, `docs/figures/**`. Achieved via mkdocs `not_in_nav:`.

The research-log and the long-form phase4_ablation.md are kept available (link-resolvable) but not surfaced in nav. They're internal-feeling for the engineering-due-diligence audience.

## 4. Tech stack

Tooling is already in `pyproject.toml`'s `docs` optional extra:

- `mkdocs>=1.6`
- `mkdocs-material>=9.5`
- `mkdocstrings[python]>=0.25`

No new dependencies. CI gets a new `.[docs]` install step in the docs workflow only.

## 5. `mkdocs.yml` configuration

Single repo-root `mkdocs.yml`. Key decisions:

- `theme.name: material` with `navigation.tabs`, `navigation.sections`, `navigation.expand`, `navigation.top`, `search.highlight`, `search.suggest`, `content.code.copy`, `content.action.edit`.
- Palette: light + dark with a toggle, primary colour `indigo`. No custom logo or icon.
- `repo_url: https://github.com/RDechua/roboeval`, `edit_uri: edit/main/docs/` so every page gets a pencil link back to source.
- `plugins: [search, mkdocstrings]` — mkdocstrings configured with the Google docstring style (matches the codebase), `show_source: true`, `members_order: source`, `separate_signature: true`.
- Markdown extensions: `admonition`, `pymdownx.details`, `pymdownx.superfences` with the mermaid custom fence so the blog's residual diagram renders, `pymdownx.tabbed`, `tables`, `toc: { permalink: true }`.
- Explicit `nav:` block declaring the 4-section structure exactly.
- `not_in_nav:` excludes the internal docs listed above.
- `extra.social:` links to GitHub and the HF Spaces dashboard.

## 6. New pages

| File | Purpose | Approx size |
|---|---|---|
| `docs/index.md` | Landing page. One paragraph pitch, three feature cards or a small linked list pointing at Live Demo, Blog, API. | ~150 words + card block |
| `docs/getting-started.md` | Promoted README quickstart. `uv venv .venv`, `uv pip install -e '.[dev]'`, `roboeval smoke`, `roboeval dashboard run` — exact commands a fresh clone needs. | ~80 lines |
| `docs/reference/index.md` | API overview: one paragraph + a table linking the 7 submodule pages with one-line "what's in here" descriptions. | ~60 words + table |
| `docs/reference/cli.md` | `# CLI` + `::: roboeval.cli`. | ~6 lines |
| `docs/reference/envs.md` | `# Environments` + `::: roboeval.envs`. | ~6 lines |
| `docs/reference/evaluation.md` | `# Evaluation` + `::: roboeval.evaluation`. | ~6 lines |
| `docs/reference/policies.md` | `# Policies` + `::: roboeval.policies`. | ~6 lines |
| `docs/reference/taxonomy.md` | `# Failure-mode taxonomy` + `::: roboeval.taxonomy`. | ~6 lines |
| `docs/reference/residual.md` | `# Residual RL` + `::: roboeval.residual`. | ~6 lines |
| `docs/reference/dashboard.md` | `# Dashboard` + `::: roboeval.dashboard`. | ~6 lines |

Each `:::` directive recurses through the module — mkdocstrings handles file-by-file submodule rendering.

## 7. CI / deploy

New workflow: `.github/workflows/docs.yml`.

Triggers on push to `main` that touches `docs/**`, `roboeval/**`, `mkdocs.yml`, or `pyproject.toml`.

Steps:

1. `actions/checkout@v4`
2. `actions/setup-python@v5` with Python 3.11, pip cache
3. `pip install --upgrade pip && pip install -e '.[docs]'`
4. `mkdocs build --strict` — fails the workflow on any broken link, missing nav target, or unknown plugin option. This is the docs equivalent of `mypy --strict`.
5. `mkdocs gh-deploy --force --remote-branch gh-pages` — pushes the built site to the `gh-pages` branch.

GitHub Pages is configured (one-time, via repo settings → Pages) to serve from `gh-pages` branch root. URL becomes `https://rdechua.github.io/roboeval/`.

The existing `ci.yml` is untouched — docs deploy is its own workflow with `contents: write` permission for the gh-deploy step.

## 8. Testing

`tests/docs/test_mkdocs_build.py` — one test that calls `subprocess.run([sys.executable, "-m", "mkdocs", "build", "--strict", "--site-dir", str(tmp_path)])` and asserts exit code 0. Gated with `pytest.importorskip("mkdocs")` so the existing ci.yml (which does not install `.[docs]`) stays green; the new docs.yml workflow installs `.[docs]` so it exercises the test.

No prose-level testing. Markdown content is reviewed by the user during the per-page commits.

## 9. Edge cases & error handling

- **Mermaid in the blog post** — handled by the `pymdownx.superfences` mermaid custom fence in `mkdocs.yml`. Without that block, mermaid renders as plain code and the architecture diagram in the blog post breaks.
- **Relative image links in the blog post** (`../figures/...`) — mkdocs resolves them correctly because both `docs/blog/` and `docs/figures/` live under `docs/`.
- **Broken external links to the HF Space** — out of scope for `--strict`; mkdocs-strict only checks internal links.
- **Auto-deploy race** — if two commits land in quick succession on `main` that both touch docs paths, the second deploy job will overwrite the first. Acceptable.

## 10. Acceptance

- [ ] `mkdocs serve` renders the site locally; no warnings at default verbosity.
- [ ] `mkdocs build --strict` exits 0 (no broken links / unknown nav targets).
- [ ] `.github/workflows/docs.yml` runs green on push to `main` and publishes to `gh-pages`.
- [ ] `https://rdechua.github.io/roboeval/` loads and shows 4 top-level tabs.
- [ ] Every `roboeval.<submodule>` mkdocstrings page renders at least one public function (PRD §9.1 "all public functions documented").
- [ ] The Getting Started commands work on a fresh clone with no modification (PRD §9.1 acceptance).
- [ ] Top-level `README.md` adds a "Docs" badge alongside the existing Live Demo + Honest Null badges.
- [ ] `docs/STATE.md` marks the MkDocs deliverable closed.

## 11. Out of scope (deferred)

- Custom domain (`docs.roboeval.io` or similar). Subdomain `rdechua.github.io/roboeval/` is fine for v1.0.
- Custom logo / colour palette beyond the indigo default. Easy v1.1 polish if the rest of the visual identity converges.
- Versioned docs (mike). Only one version exists; multi-version routing is overkill until the public API drifts.
- Search beyond the default plugin. mkdocs-material's built-in search is good.
- Translations / i18n.
- The deferred items from prior specs (HF Blog cross-post, arXiv PDF) remain deferred.
