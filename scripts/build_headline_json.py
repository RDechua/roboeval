"""Build the tracked ``data/headline.json`` artifact for the Phase 5 dashboard.

Reads ``data/taxonomy/auto_labels_<run_id>.json`` files (gitignored,
regeneratable from W&B) plus the per-cell mean/std from ``docs/STATE.md``
to produce ``data/headline.json``, a single committed artifact that the
dashboard loads at runtime.

Per-cell ``mean_tsr_custom`` and ``std_tsr_custom`` for the 10 Phase 3
cells come from ``docs/STATE.md`` and are hard-coded here for traceability
(STATE.md is the human-readable source of truth; this script mechanises
the transcription).

Usage::

    python -m scripts.build_headline_json
    # writes data/headline.json
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
from pathlib import Path
from typing import Any

_LOG = logging.getLogger("scripts.build_headline_json")

# (mean_tsr_custom, std_tsr_custom) per Phase 3 cell, transcribed from
# docs/STATE.md "Spatial degradation curve" and "Temporal degradation".
_PHASE3_STATS: dict[str, tuple[float, float]] = {
    # spatial
    "y-5cm": (0.127, 0.009),
    "y-3cm": (0.553, 0.025),
    "y-1cm": (0.827, 0.034),
    "y+1cm": (0.720, 0.102),
    "y+3cm": (0.553, 0.041),
    "y+5cm": (0.307, 0.019),
    # nominal anchor
    "nominal": (0.800, 0.057),
    # temporal
    "delay-1step": (0.753, 0.050),
    "delay-3step": (0.767, 0.068),
    "delay-5step": (0.687, 0.066),
}

# Map config_path basenames to canonical cell_ids. Fresh evals write
# the config_path; W&B-relabeled artifacts store a ``<wandb:...>``
# placeholder, so this map is only the fast path — the
# perturbation-params fallback handles the rest.
_CONFIG_TO_CELL: dict[str, str] = {
    "act_spatial_y-5cm.yaml": "y-5cm",
    "act_spatial_y-3cm.yaml": "y-3cm",
    "act_spatial_y-1cm.yaml": "y-1cm",
    "act_spatial_y+1cm.yaml": "y+1cm",
    "act_spatial_y+3cm.yaml": "y+3cm",
    "act_spatial_y+5cm.yaml": "y+5cm",
    "act_nominal.yaml": "nominal",
    # Temporal configs use the singular "step" suffix in the repo:
    "act_temporal_delay_1step.yaml": "delay-1step",
    "act_temporal_delay_3step.yaml": "delay-3step",
    "act_temporal_delay_5step.yaml": "delay-5step",
    # Defensive: in case a future config drops the trailing "s".
    "act_temporal_delay_1steps.yaml": "delay-1step",
    "act_temporal_delay_3steps.yaml": "delay-3step",
    "act_temporal_delay_5steps.yaml": "delay-5step",
}

# Magnitude in each cell's natural units (cm for spatial, steps for
# temporal, 0.0 for nominal).
_MAGNITUDES: dict[str, float] = {
    "y-5cm": -0.05,
    "y-3cm": -0.03,
    "y-1cm": -0.01,
    "y+1cm": 0.01,
    "y+3cm": 0.03,
    "y+5cm": 0.05,
    "nominal": 0.0,
    "delay-1step": 1.0,
    "delay-3step": 3.0,
    "delay-5step": 5.0,
}


def _axis_for_cell(cell_id: str) -> str:
    if cell_id == "nominal":
        return "nominal"
    if cell_id.startswith("delay-"):
        return "temporal"
    return "spatial"


def _cell_from_perturbation(payload: dict[str, Any]) -> str | None:
    """Infer the cell_id from a payload's perturbation_kind + params.

    Falls back to this when ``config_path`` is a ``<wandb:...>``
    placeholder (W&B-relabeled artifacts) instead of a tracked YAML.
    """
    kind = payload.get("perturbation_kind")
    params = payload.get("perturbation_params") or {}
    if kind == "none" or (kind == "spatial" and not params):
        return "nominal"
    if kind == "spatial":
        dx = float(params.get("dx_m", 0.0))
        dy = float(params.get("dy_m", 0.0))
        if dx != 0:
            return None
        cm = round(dy * 100)
        if cm == 0:
            return "nominal"
        sign = "+" if cm > 0 else "-"
        return f"y{sign}{abs(cm)}cm"
    if kind == "temporal":
        steps = int(params.get("delay_steps", params.get("n_steps", 0)))
        if steps <= 0:
            return None
        return f"delay-{steps}step"
    return None


def _payload_is_residual(payload: dict[str, Any]) -> bool:
    """Identify Phase 4 residual runs we want to exclude from headline cells."""
    cfg = str(payload.get("config_path", ""))
    return "residual" in cfg


def _has_real_config_path(payload: dict[str, Any]) -> bool:
    """A payload has a "real" config path if it points at a tracked YAML."""
    cfg = str(payload.get("config_path", ""))
    return cfg.startswith("configs/") and cfg.endswith(".yaml")


def _payload_is_pytest_fixture(payload: dict[str, Any]) -> bool:
    """Return ``True`` for auto_labels written by pytest tmpdir fixtures.

    Those payloads have absolute ``/tmp`` paths or a ``cli_test.yaml``
    basename — never real Phase 3 evals.
    """
    cfg = str(payload.get("config_path", ""))
    if not cfg:
        return False
    if cfg.startswith("/"):
        return True
    if "pytest" in cfg:
        return True
    return Path(cfg).name == "cli_test.yaml"


def _scan_auto_labels(repo_root: Path) -> dict[str, dict[str, Any]]:
    """Return ``{cell_id: auto_labels_payload}``.

    Prefers payloads with a tracked ``config_path`` when multiple
    candidates exist for the same cell (e.g., a fresh eval plus a
    W&B-relabel sample).
    """
    by_cell: dict[str, dict[str, Any]] = {}
    labels_dir = repo_root / "data" / "taxonomy"
    for path in sorted(labels_dir.glob("auto_labels_*.json")):
        payload = json.loads(path.read_text())
        if _payload_is_residual(payload):
            continue
        if _payload_is_pytest_fixture(payload):
            continue
        config_basename = Path(str(payload.get("config_path", ""))).name
        cell_id = _CONFIG_TO_CELL.get(config_basename)
        if cell_id is None:
            cell_id = _cell_from_perturbation(payload)
        if cell_id is None:
            # Pytest fixture leftover or unrecognised perturbation.
            continue
        if cell_id in by_cell:
            existing = by_cell[cell_id]
            keep_new = _has_real_config_path(payload) and not _has_real_config_path(
                existing
            )
            if keep_new:
                _LOG.info(
                    "preferring fresh-config payload %s for cell %s over %s",
                    payload["run_id"],
                    cell_id,
                    existing["run_id"],
                )
                by_cell[cell_id] = payload
            else:
                _LOG.info(
                    "duplicate auto_labels for cell %s: keeping %s, ignoring %s",
                    cell_id,
                    existing["run_id"],
                    payload["run_id"],
                )
            continue
        by_cell[cell_id] = payload
    return by_cell


def build_headline_payload(*, repo_root: Path) -> dict[str, Any]:
    """Produce the headline.json payload as a Python dict.

    Args:
        repo_root: Absolute path to the RoboEval repository root.

    Returns:
        The headline.json payload (schema_version 1) ready to be
        ``json.dumps``-ed.

    Raises:
        FileNotFoundError: when expected auto_labels or ablation files
            are missing from disk.
    """
    auto_labels = _scan_auto_labels(repo_root)
    missing = sorted(set(_PHASE3_STATS) - set(auto_labels))
    if missing:
        raise FileNotFoundError(
            f"missing auto_labels for cells {missing!r}; "
            f"regenerate via scripts/relabel_from_wandb.py"
        )

    cells: list[dict[str, Any]] = []
    for cell_id, (mean, std) in _PHASE3_STATS.items():
        payload = auto_labels[cell_id]
        distribution = payload["distribution"]
        n_rollouts = sum(distribution.values())
        cells.append(
            {
                "cell_id": cell_id,
                "axis": _axis_for_cell(cell_id),
                "magnitude": _MAGNITUDES[cell_id],
                "mean_tsr_custom": mean,
                "std_tsr_custom": std,
                "per_seed_tsr_custom": None,
                "mean_tsr": None,
                "median_tts": None,
                "failure_counts": distribution,
                "n_rollouts": n_rollouts,
                "run_id": payload["run_id"],
            }
        )

    return {
        "schema_version": 1,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        "source": (
            "Built by scripts/build_headline_json.py from "
            "data/taxonomy/auto_labels_*.json + docs/STATE.md."
        ),
        "cells": cells,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    payload = build_headline_payload(repo_root=repo_root)
    out_path = repo_root / "data" / "headline.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    _LOG.info("wrote %s (%d cells)", out_path, len(payload["cells"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
