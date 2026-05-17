r"""Export a stratified, redacted relabel sample for PRD §7.3 step 4.

The blinded self-relabel protocol requires a "memory-wash" of ≥7 days
between auto-label and manual-relabel. To enforce blinding, the manual
labeller works from a *redacted* sample — rollout identifiers only, no
auto-labels visible. After ≥7 days they record their independent labels
against the same rollouts; ``cohens_kappa(auto, manual)`` then
quantifies agreement (target κ > 0.6).

This script reads one ``auto_labels_<run_id>.json`` (frozen artifact
from ``roboeval evaluate`` or ``relabel_from_wandb.py``) and writes:

* ``relabel_sample_<run_id>.json`` — identifiers for N rollouts per
  failure-mode bucket (stratified). Auto-labels are NOT included.
  Hash of the source auto_labels JSON is stored for chain-of-custody.
* ``relabel_unlock_at_<run_id>.txt`` — ISO-8601 timestamp ≥7 days
  hence. The labelling UI reads this file and refuses to display
  the sample until ``now() >= unlock_at``.

Sampling: per-category stratified, deterministic given ``(run_id, seed)``
so two runs of this script against the same input produce the same
sample. If a category has fewer rollouts than the requested N, the
script takes all rollouts in that category (and logs a warning to
stderr).

Usage
-----
::

    python scripts/export_relabel_sample.py \\
        --input data/taxonomy/auto_labels_cm6uf89g.json \\
        --per-category-n 5 \\
        --out data/taxonomy

Schema (v1)
-----------
::

    {
      "schema_version": 1,
      "auto_labels_source": "data/taxonomy/auto_labels_<run_id>.json",
      "auto_labels_sha256": "<hex>",
      "run_id": "<run_id>",
      "per_category_n": 5,
      "exported_at": "2026-05-17T20:00:00Z",
      "unlock_at": "2026-05-24T20:00:00Z",
      "samples": [
        {"seed_group": 0, "rollout_idx": 17, "episode_seed": 17,
         "manual_failure_mode": null},
        ...
      ]
    }
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION: int = 1
DEFAULT_PER_CATEGORY_N: int = 5
DEFAULT_LOCK_DAYS: int = 7
"""PRD §7.3 step 4 explicitly: 'wait >= 7 days for a memory-wash'."""

_REDACTED_SENTINEL: None = None
"""Value used for the manual-failure-mode field at export time."""


def _sha256_of_path(path: Path) -> str:
    """Hex SHA-256 of a file's contents, for the chain-of-custody entry."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_sample(
    labels: list[Mapping[str, Any]],
    per_category_n: int,
    *,
    seed: int,
) -> list[Mapping[str, Any]]:
    """Pick N rollouts per failure-mode bucket, deterministically.

    Groups labels by ``failure_mode`` (``null`` is the "success" bucket),
    shuffles within each bucket with a seeded RNG, takes the first N. If
    a bucket has fewer than N entries, all of them are taken.

    Args:
        labels: ``labels`` list from an ``auto_labels_*.json`` file.
        per_category_n: Target count per bucket.
        seed: RNG seed. The script uses
            ``int.from_bytes(sha256(run_id), 'big') % 2**32`` so two
            invocations against the same input produce the same sample.

    Returns:
        Combined sample list, ordered by ``(failure_mode, seed_group,
        rollout_idx)``. Length is at most ``per_category_n * #buckets``.
    """
    by_bucket: dict[str | None, list[Mapping[str, Any]]] = defaultdict(list)
    for label in labels:
        by_bucket[label.get("failure_mode")].append(label)

    rng = random.Random(seed)
    picked: list[Mapping[str, Any]] = []
    for bucket in sorted(by_bucket, key=lambda x: ("" if x is None else x)):
        bucket_labels = list(by_bucket[bucket])
        rng.shuffle(bucket_labels)
        chosen = bucket_labels[:per_category_n]
        if len(bucket_labels) < per_category_n:
            print(
                f"warning: bucket {bucket!r} has only {len(bucket_labels)} "
                f"rollouts (< requested {per_category_n}); taking all.",
                file=sys.stderr,
            )
        picked.extend(chosen)

    # Final sort for stable output ordering.
    picked.sort(
        key=lambda r: (
            "" if r.get("failure_mode") is None else r.get("failure_mode", ""),
            int(r.get("seed_group", 0)),
            int(r.get("rollout_idx", 0)),
        )
    )
    return picked


def build_sample_obj(
    auto_labels_obj: Mapping[str, Any],
    auto_labels_path: Path,
    *,
    per_category_n: int,
    lock_days: int,
    now: datetime,
) -> dict[str, Any]:
    """Construct the schema-v1 relabel-sample JSON object.

    Pure-function — no I/O — so it round-trips cleanly in tests.

    Args:
        auto_labels_obj: Parsed ``auto_labels_<run_id>.json`` dict.
        auto_labels_path: Path to the source file (used for the
            ``auto_labels_source`` reference + SHA-256 audit).
        per_category_n: Target sample size per failure-mode bucket.
        lock_days: Days from ``now`` until the sample is unlocked
            for manual labelling (PRD §7.3 step 4 minimum is 7).
        now: Wall clock at export time (injectable for tests).

    Returns:
        JSON-serializable dict matching the v1 schema.
    """
    run_id = str(auto_labels_obj.get("run_id", "unknown"))
    labels = list(auto_labels_obj.get("labels", []))

    seed_bytes = hashlib.sha256(run_id.encode("utf-8")).digest()[:4]
    seed = int.from_bytes(seed_bytes, "big")

    chosen = stratified_sample(labels, per_category_n, seed=seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "auto_labels_source": str(auto_labels_path),
        "auto_labels_sha256": _sha256_of_path(auto_labels_path),
        "run_id": run_id,
        "per_category_n": per_category_n,
        "exported_at": now.isoformat(),
        "unlock_at": (now + timedelta(days=lock_days)).isoformat(),
        "samples": [
            {
                "seed_group": int(label.get("seed_group", 0)),
                "rollout_idx": int(label.get("rollout_idx", 0)),
                "episode_seed": int(label.get("episode_seed", 0)),
                "manual_failure_mode": _REDACTED_SENTINEL,
            }
            for label in chosen
        ],
    }


def export_relabel_sample(
    input_path: Path,
    output_dir: Path,
    *,
    per_category_n: int = DEFAULT_PER_CATEGORY_N,
    lock_days: int = DEFAULT_LOCK_DAYS,
    now: datetime | None = None,
) -> tuple[Path, Path]:
    """Read auto_labels → write redacted sample + unlock_at sidecar.

    Args:
        input_path: Path to ``auto_labels_<run_id>.json``.
        output_dir: Directory to write the sample + unlock_at file into.
        per_category_n: Target count per failure-mode bucket.
        lock_days: Days to lock the sample.
        now: Wall clock; defaults to ``datetime.now(timezone.utc)``.

    Returns:
        ``(sample_path, unlock_path)`` — paths to the two written files.
    """
    if now is None:
        now = datetime.now(UTC)
    auto_obj = json.loads(input_path.read_text())
    sample_obj = build_sample_obj(
        auto_obj,
        input_path,
        per_category_n=per_category_n,
        lock_days=lock_days,
        now=now,
    )
    run_id = str(sample_obj["run_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / f"relabel_sample_{run_id}.json"
    unlock_path = output_dir / f"relabel_unlock_at_{run_id}.txt"
    sample_path.write_text(json.dumps(sample_obj, indent=2))
    unlock_path.write_text(str(sample_obj["unlock_at"]) + "\n")
    return sample_path, unlock_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to a frozen auto_labels_<run_id>.json artifact.",
    )
    parser.add_argument(
        "--per-category-n",
        type=int,
        default=DEFAULT_PER_CATEGORY_N,
        help=f"Sample size per failure-mode bucket (default {DEFAULT_PER_CATEGORY_N}).",
    )
    parser.add_argument(
        "--lock-days",
        type=int,
        default=DEFAULT_LOCK_DAYS,
        help=f"Days to lock the sample (PRD min 7, default {DEFAULT_LOCK_DAYS}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/taxonomy"),
        help="Output directory (default: data/taxonomy).",
    )
    args = parser.parse_args(argv)
    try:
        sample_path, unlock_path = export_relabel_sample(
            args.input,
            args.out,
            per_category_n=args.per_category_n,
            lock_days=args.lock_days,
        )
    except Exception as exc:  # noqa: BLE001 - script-level boundary
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {sample_path}")
    print(f"Wrote {unlock_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
