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
