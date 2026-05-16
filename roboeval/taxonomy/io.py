"""Frozen auto-labels artifact writer for the PRD §7.3 protocol.

Writes the rule-based classifier output as a JSON file under
``data/taxonomy/auto_labels_<run_id>.json``. The file is the evidence
trail that the blinded self-relabel (PRD §7.3 step 4) reads ≥7 days
later; the schema is versioned so future relabel runs can match against
the exact classifier rules that produced each label.

Schema (v1)
-----------
::

    {
      "schema_version": 1,
      "run_id": "<wandb-run-id-or-timestamp>",
      "config_path": "configs/.../act_spatial_y+1cm.yaml",
      "policy_id": "lerobot/act_aloha_sim_transfer_cube_human",
      "env_id": "gym_aloha/AlohaTransferCube-v0",
      "perturbation_kind": "spatial" | "none",
      "perturbation_params": {...},
      "perturbation_applied": true | false,
      "n_rollouts": 150,
      "distribution": {"success": 108, "timeout": 30, ...},
      "labels": [
        {
          "seed_group": 0,
          "rollout_idx": 0,
          "episode_seed": 0,
          "failure_mode": null | "grasp_failure" | ...,
          "evidence": {...}
        },
        ...
      ]
    }

The ``failure_mode`` value is ``null`` for primary-successful rollouts
and one of the :class:`FailureMode` enum string values otherwise.
``distribution`` uses ``"success"`` as the key for the ``null`` bucket
so all keys are JSON strings.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from roboeval.taxonomy.types import FailureMode, RolloutLabel

SCHEMA_VERSION: int = 1
"""Bumped on any breaking change to the on-disk JSON shape."""

_SUCCESS_BUCKET_KEY: str = "success"
"""Distribution key for primary-successful rollouts (``failure_mode=None``)."""


def compute_distribution(labels: Sequence[RolloutLabel]) -> dict[str, int]:
    """Count rollouts per failure-mode bucket.

    Includes a zero count for every :class:`FailureMode` value that
    didn't appear in ``labels``, so downstream dashboards never have to
    guess whether a missing key means "zero" or "never measured".

    Args:
        labels: One :class:`RolloutLabel` per rollout.

    Returns:
        Dict mapping ``"success"`` plus each :class:`FailureMode` value
        to its count. Sum of values equals ``len(labels)``.
    """
    counts: Counter[str] = Counter()
    for label in labels:
        if label.failure_mode is None:
            counts[_SUCCESS_BUCKET_KEY] += 1
        else:
            counts[label.failure_mode.value] += 1
    out: dict[str, int] = {_SUCCESS_BUCKET_KEY: counts.get(_SUCCESS_BUCKET_KEY, 0)}
    for mode in FailureMode:
        out[mode.value] = counts.get(mode.value, 0)
    return out


def labels_to_json_obj(
    labels: Sequence[RolloutLabel],
    *,
    run_id: str,
    config_path: str,
    policy_id: str,
    env_id: str,
    perturbation_kind: str,
    perturbation_params: Mapping[str, Any],
    perturbation_applied: bool,
) -> dict[str, Any]:
    """Build the schema-v1 JSON object for an auto-labels artifact.

    Pure data — no I/O. Used by :func:`write_auto_labels` and the unit
    tests that round-trip the schema.

    Args:
        labels: Per-rollout classifier outputs.
        run_id: Stable identifier (W&B run id or timestamped run name)
            used in the artifact filename.
        config_path: Path to the eval YAML config that produced these
            rollouts; recorded so reviewers can re-instantiate the run.
        policy_id: HF repo id of the evaluated policy.
        env_id: Gymnasium env id.
        perturbation_kind: ``"none"`` or the perturbation kind from the
            config (``spatial`` / ``visual`` / ``dynamic`` / ``temporal``).
        perturbation_params: The perturbation kwargs dict.
        perturbation_applied: Whether any perturbation was applied. Drives
            the Recovery-failure rule and is the run-level filter the
            heatmap groups by.

    Returns:
        JSON-serializable dict matching the v1 schema.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_path": config_path,
        "policy_id": policy_id,
        "env_id": env_id,
        "perturbation_kind": perturbation_kind,
        "perturbation_params": dict(perturbation_params),
        "perturbation_applied": perturbation_applied,
        "n_rollouts": len(labels),
        "distribution": compute_distribution(labels),
        "labels": [
            {
                "seed_group": label.seed_group,
                "rollout_idx": label.rollout_idx,
                "episode_seed": label.episode_seed,
                "failure_mode": (
                    None if label.failure_mode is None else label.failure_mode.value
                ),
                "evidence": dict(label.evidence),
            }
            for label in labels
        ],
    }


def write_auto_labels(
    labels: Sequence[RolloutLabel],
    output_dir: Path | str,
    *,
    run_id: str,
    config_path: str,
    policy_id: str,
    env_id: str,
    perturbation_kind: str,
    perturbation_params: Mapping[str, Any],
    perturbation_applied: bool,
) -> Path:
    """Write the auto-labels JSON to ``<output_dir>/auto_labels_<run_id>.json``.

    Creates the parent directory if missing. The file is **frozen** by
    PRD §7.3 step 4 — callers should treat the path as write-once. Use
    a distinct ``run_id`` per evaluation to avoid clobbering.

    Args:
        labels: Per-rollout classifier outputs.
        output_dir: Directory to write into. Created with ``parents=True``
            if it doesn't exist; PRD convention is ``data/taxonomy``.
        run_id: See :func:`labels_to_json_obj`.
        config_path: See :func:`labels_to_json_obj`.
        policy_id: See :func:`labels_to_json_obj`.
        env_id: See :func:`labels_to_json_obj`.
        perturbation_kind: See :func:`labels_to_json_obj`.
        perturbation_params: See :func:`labels_to_json_obj`.
        perturbation_applied: See :func:`labels_to_json_obj`.

    Returns:
        Path to the written JSON file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"auto_labels_{run_id}.json"
    obj = labels_to_json_obj(
        labels,
        run_id=run_id,
        config_path=config_path,
        policy_id=policy_id,
        env_id=env_id,
        perturbation_kind=perturbation_kind,
        perturbation_params=perturbation_params,
        perturbation_applied=perturbation_applied,
    )
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


__all__ = [
    "SCHEMA_VERSION",
    "compute_distribution",
    "labels_to_json_obj",
    "write_auto_labels",
]
