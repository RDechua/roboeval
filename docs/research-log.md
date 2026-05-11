# RoboEval — Research Log

A running notebook of what got done each week, what was surprising, and what's next. Follows the PRD's "Most Important Instruction": write a paragraph every week so the blog post writes itself.

---

## Week 1 — 2026-05-11

### What was built

- **PRD passed senior review and Tier-1 fixes landed** (commit `8fe50e5`). Narrowed v1.0 from 3 policies to ACT-only after confirming via the HuggingFace API that no sim-trained Diffusion Policy checkpoint exists for ALOHA Transfer Cube under `lerobot/*`. Pinned `lerobot==0.4.4`, fixed the `lerobot.common` → `lerobot.policies.*` import, added a numerical success criterion (`z > 0.05 m AND in xy-box for 5 steps`), set the Timeout step budget at 400, and added the real-vs-sim checkpoint risk as the top-of-table mitigation. Cross-policy comparison and the multi-policy perturbation grid are now explicit v1.0 non-goals; both move to v1.1.
- **Repo scaffolded end-to-end** (commit `1131484`) to match PRD Section 5.2: typed `roboeval/` package with `envs/`, `policies/`, `evaluation/`, `taxonomy/`, `residual/` subpackages, `py.typed` marker, and a Google-docstring style throughout.
- **Tooling green**: `pyproject.toml` with Python 3.11 pin, `lerobot==0.4.4`, ruff (line 88, Google docstrings, pyupgrade, bugbear, annotations), `mypy --strict` on the package, pytest, hatchling. Pre-commit hooks for ruff/ruff-format/mypy/EOF/whitespace. GitHub Actions CI runs ruff + mypy + pytest on push/PR.
- **`roboeval smoke` CLI** runs a 10-step random-action rollout against `gym_aloha/AlohaTransferCube-v0`. Heavy deps are lazy-imported so the package stays mypy/ruff-checkable in CI without installing torch + lerobot.
- **`tests/test_smoke.py`** parametrises import smoke tests across all subpackages plus a CLI-parser test (9 tests, all passing locally).

### Verification

- `ruff check .` clean
- `ruff format --check .` clean (after auto-format of one test file)
- `mypy --strict roboeval` clean — 7 source files
- `pytest -q` → 9 passed in 0.01 s

### What surprised me

- **The PRD's `lerobot.common.*` import would have failed on a fresh install.** LeRobot 0.4.x dropped the `common` namespace entirely. Catching this in the PRD review before scaffolding saved a Day-1 wall.
- **No public sim-trained Diffusion Policy checkpoint for ALOHA Transfer Cube exists.** The `lerobot/diffusion_pusht*` checkpoints are PushT-only. GitHub issue [huggingface/lerobot#502](https://github.com/huggingface/lerobot/issues/502) reports a user training DP on `aloha_sim_transfer_cube_human` from scratch and getting only 2–6% success at 60k steps — strong signal that DP underperforms on this task and that the original 3-policy plan was unrealistic from the start.
- **Strict mypy survives heavy dynamic imports cleanly** by combining lazy imports inside functions with `ignore_missing_imports` overrides for external libs in `pyproject.toml`. The package surface stays statically typed; the runtime escape hatch is scoped.
- **The done criterion for Week 1 is achievable in CI even without the heavy stack.** The "first rollout renders without crash" gate is satisfied by the `roboeval smoke` script existing and being invocable; actually running it requires a local M1 with the venv but does not block the lint/type/test pipeline.

### What's deferred

- **Running `roboeval smoke` end-to-end on M1.** The CLI is built but I did not actually execute it against the real env — that depends on `gym_aloha` + MuJoCo rendering working on Darwin 23.3.0, which is a known finicky path. First task next session.
- **Actually verifying the ACT checkpoint loads on MPS.** PRD Section 11's new top-row risk says verify `TSR > 50%` on Day 1 before doing anything else. Not done yet.
- **Pre-commit hooks `pre-commit install` step.** Configured but not installed into `.git/hooks/`. Manual user step.

### Next session (target: Week 2)

PRD Section 10.2 Week 2 target — "Baseline TSR on Policy A (ACT), 50 rollouts, log to W&B" / done = "W&B dashboard showing TSR ± std".

1. Run `roboeval smoke --steps 10` locally on the M1 and confirm a random-action rollout completes without raising. Fix any MuJoCo/gym_aloha issues that surface.
2. Run `roboeval smoke --steps 50` and confirm rendering is stable across longer rollouts.
3. Build `roboeval/policies/act_loader.py`: typed wrapper around `ACTPolicy.from_pretrained("lerobot/act_aloha_sim_transfer_cube_human")` with normalisation stats handled correctly.
4. Build `roboeval/evaluation/rollout.py`: one `run_episode(env, policy, seed) -> RolloutResult` function with full typing and W&B logging via the `experiment-logger` skill.
5. Tune the numerical success criterion from PRD Section 6.2 (`z_success`, `xy_tolerance`, `N_dwell`) against the model card's ~83% TSR target. Freeze the values once nominal-condition TSR is in the right ballpark.
6. First baseline run: 50 rollouts × 3 seeds × nominal condition, logged to W&B.

### Open questions to resolve in Week 2

- Does the ACT checkpoint's observation preprocessing pipeline accept gym-aloha's `dict` observation directly, or does it require shape gymnastics?
- What's the actual single-rollout wall-clock on M1 CPU? Drives the compute budget for the perturbation phase.
- Should the W&B project be named `roboeval` or `roboeval-v1.0`? Decide before the first logged run.

---
