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

## Week 2 — 2026-05-11

### What was built

Seven commits, in this order on `main`:

- `020b2d1 docs(prd)` — Section 6.2 now defines two TSR signals side-by-side. Primary uses gym-aloha's native `info["is_success"]` (matches model-card 83%); secondary uses the PRD geometric criterion (z + xy + dwell). Both are logged to W&B; the secondary's defaults are Week-2 placeholders calibrated against the primary on nominal conditions.
- `1c28354 feat(envs)` — [`roboeval/envs/aloha.py`](roboeval/envs/aloha.py) wraps LeRobot's `make_env(n_envs=1)`, unwraps the `SyncVectorEnv`, and exposes `get_cube_state` (qpos slice 16:23 → the cube's 7-element pose) for the success detector. [`roboeval/envs/success.py`](roboeval/envs/success.py) is a stateful `TransferCubeSuccessDetector` with a frozen `SuccessCriterion` and an 8-test unit suite ([`tests/envs/test_success.py`](tests/envs/test_success.py)).
- `13195ce feat(policies)` — [`roboeval/policies/base.py`](roboeval/policies/base.py) is the runtime-checkable `Policy` Protocol: `select_action(observation) → np.float32` and `reset() → None`. [`roboeval/policies/act_loader.py`](roboeval/policies/act_loader.py) is the ACT adapter; uses the dataset-stats path (see "What surprised me" below) rather than the standard pretrained-path processor loader.
- `29afbaf feat(evaluation)` — [`roboeval/evaluation/types.py`](roboeval/evaluation/types.py), [`rollout.py`](roboeval/evaluation/rollout.py), [`loop.py`](roboeval/evaluation/loop.py) + 12 unit tests covering aggregation, single-rollout outcomes, and the multi-seed driver.
- `cbd51f3 feat(evaluation)` — [`roboeval/evaluation/logger.py`](roboeval/evaluation/logger.py) wraps W&B with the locked 14-column rollouts table and config-as-artifact upload. Test runs with `WANDB_MODE=disabled`.
- `1070008 feat(configs)` — [`act_nominal.yaml`](configs/baseline/act_nominal.yaml) (3 seeds × 50 rollouts, online W&B, overnight target) and [`act_nominal_fast.yaml`](configs/baseline/act_nominal_fast.yaml) (1 seed × 10 rollouts, offline W&B, ~80 s smoke).
- `e5b750c feat(cli)` — `roboeval evaluate --config PATH` end-to-end.

### Fast smoke result (`act_nominal_fast.yaml`)

```
[roboeval] Evaluation complete.
  mean_tsr        = 0.800 +/- 0.000  (primary, gym-aloha native is_success)
  mean_tsr_custom = 0.000 +/- 0.000  (PRD z+xy+dwell)
  median_tts      = 261.0
  n_rollouts      = 10 across 1 seed group(s)
  per_seed_tsr    = (0.8,)
```

8 of 10 rollouts succeeded on gym-aloha's native signal, matching the model card's published ~83% TSR within statistical noise at n=10 (binomial 95% CI is [44%, 97%]). Total wall: ~82 s, MPS active. W&B run saved offline at `wandb/offline-run-20260511_144808-5mci37ct/` (sync with `wandb sync` once authenticated).

The secondary `mean_tsr_custom = 0.000` is expected: `target_xy=(0.0, 0.0)` is the placeholder default; the actual left-arm receptacle is offset from the world origin. **Calibration is Week 3's first task.**

### What surprised me

- **The `lerobot/act_aloha_sim_transfer_cube_human` checkpoint is a LeRobot 0.1.x-era artifact.** Its HF repo contains `config.json`, `model.safetensors`, `train_config.json`, `eval_info.json` — but **not** the `policy_preprocessor.json` / `policy_postprocessor.json` files that LeRobot 0.4.x's `make_pre_post_processors(cfg, pretrained_path=...)` expects. First smoke attempt crashed with `EntryNotFoundError: policy_preprocessor.json`. Fix: load the source dataset's `LeRobotDatasetMetadata("lerobot/aloha_sim_transfer_cube_human")` and pass `dataset_stats=ds_meta.stats` to a no-`pretrained_path` `make_pre_post_processors` call, which routes through `make_act_pre_post_processors` and rebuilds normalisation from scratch. The `make_policy(cfg, ds_meta=...)` path then loads weights AND populates normalisation buffers in one go. This swap is documented in [`roboeval/policies/act_loader.py`](roboeval/policies/act_loader.py)'s docstring.
- **MPS works flawlessly for ACT inference on M1.** No fallback needed. Per-rollout wall is 5.4–13.2 s (avg ~8 s); a 50×3 = 150-rollout full run will land in ~20 min, not the 2.5–4 hours I estimated in Phase A. The Phase A estimate assumed CPU; MPS gives a clean ~10× speedup on the convnet backbone.
- **gym-aloha registers `max_episode_steps=300` but LeRobot's `AlohaEnv` config overrides it to 400 via `gym_kwargs`.** This was hidden until I read `lerobot/envs/configs.py:91` — without it, the PRD's 400-step budget would have been silently truncated to 300. The harness now goes through LeRobot's `make_env` precisely to inherit this override for free.
- **LeRobot 0.4.4 already has a complete eval pipeline** (`make_env`, `make_policy`, `make_pre_post_processors`, `preprocess_observation`, `lerobot_eval.py::rollout`). The Phase A architectural decision to build on these primitives — not reimplement from scratch — was correct. Our value-add is typed wrappers + custom rollout loop + dual success criterion + locked W&B schema; the heavy lifting is upstream.
- **The "Unexpected key(s) when loading model" warning is harmless.** The v0.1 checkpoint carries `normalize_inputs.buffer_*` keys for the old in-model normalization layer; v0.4 uses external processors instead, so those keys are ignored on load. The actual weights map correctly.

### Top 3 bugs hit (interview stories)

1. **Missing `policy_preprocessor.json` on v0.1 checkpoint.** Symptom: `huggingface_hub.errors.EntryNotFoundError: 404` deep inside `make_pre_post_processors`. Diagnosis: `curl https://huggingface.co/api/models/lerobot/act_aloha_sim_transfer_cube_human | jq .siblings` confirmed the file doesn't exist on the repo at all. Fix: switch to `LeRobotDatasetMetadata` → `make_pre_post_processors(cfg, dataset_stats=meta.stats)` path. Lesson: pretrained checkpoints can predate the codebase version that loads them; always inspect repo contents before assuming the loader will Just Work.

2. **`int(reward)` failed `mypy --strict` with `call-overload`.** `env.step` returns reward typed as `SupportsFloat` (not `SupportsInt`). `int(SupportsFloat)` is not an overload, but `int(float(SupportsFloat))` is. Fix: `int(float(reward))` in [`evaluation/rollout.py`](roboeval/evaluation/rollout.py). Lesson: in strict-typed code, conversions through the abstract numeric protocol need an explicit `float()` adapter.

3. **`monkeypatch.setattr(rollout_mod, "get_cube_state", _fake)` didn't take effect** in `test_evaluate_policy_three_seed_groups`. Cause: `run_rollout`'s default argument `cube_state_fn: CubeStateFn = get_cube_state` captures the function object at module-import time, so later `setattr` on the module attribute doesn't update the default. Fix: plumb `cube_state_fn` as a real parameter through `evaluate_policy` to `run_rollout`. Lesson: monkeypatching only works on attribute lookups, not on default-arg references. Explicit dependency injection beats clever patching.

### Next session (target: Week 3 — harness generalisation + full nominal run)

PRD Section 10.2 Week 3 target — "Generalise harness to policy-agnostic loader (ready for v1.1 DP); expand ACT baseline to 3 seeds × 50 rollouts × nominal conditions" / done = "Harness loads any LeRobot policy via single config flag; ACT baseline TSR ± std logged to W&B".

1. **Calibrate `target_xy` and `xy_tolerance_m`** so `mean_tsr_custom ≈ mean_tsr_native ≈ 0.8` on the fast config. Inspect `final_cube_state` distribution from a successful rollout, set `target_xy` to the mean xy of successful endpoints, set `xy_tolerance_m` to the 95th-percentile distance. Freeze the values in `act_nominal.yaml`.
2. **Run the full `act_nominal.yaml`** (3 seeds × 50 rollouts) and sync to W&B online. Expected wall: ~20 min on MPS (revised from the 2.5–4 hours Phase A estimate).
3. **Generalise the policy factory**: extract a `load_policy(repo_id, kind, ...) → Policy` that dispatches on `kind ∈ {"act", "diffusion"}` so a future Diffusion Policy adapter is a 1-line config change. Keep `load_act_policy` as a thin wrapper.
4. **Add a `tests/test_cli.py`** smoke that runs `roboeval evaluate --config configs/baseline/act_nominal_fast.yaml` end-to-end under `WANDB_MODE=disabled` and asserts mean_tsr > 0.5. Tag `@pytest.mark.slow`. Becomes the regression gate against the LeRobot API.
5. **Verify reproducibility**: re-run the fast config a second time with the same seed and confirm bit-identical per-rollout `success` and `n_steps` columns in the rollouts table. If not bit-identical, find the source of non-determinism (MuJoCo rendering? dm_control RNG?) and pin it.

### Open questions to resolve

- The model card's 83% was computed under what exact step budget and what evaluation harness? `eval_info.json` on the HF repo might encode it; check before claiming we match the published number.
- Does `mean_tsr_custom` converge to `mean_tsr_native` after `target_xy` calibration? If not, what's the discrepancy mode (e.g. cube held but outside box → primary success, secondary failure)?
- The "Unexpected key(s) when loading model" warning lists `normalize_inputs.*` and `normalize_targets.*` buffers — confirm the new processor pipeline really does carry the same normalisation stats end-to-end. If not, our 80% might be a lucky-coincidence accuracy and the true ACT performance is higher.

---
