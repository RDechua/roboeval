r"""Score Cohen's κ between manual relabels and the auto-classifier (G3 gate).

For each ``data/taxonomy/relabel_sample_<run_id>.json`` passed on the CLI:

1. Verify the embargo window has elapsed (``unlock_at`` ≤ now). Refuse to
   score before the unlock to preserve the blinding (PRD §7.3).
2. Verify every ``manual_failure_mode`` is filled in (no ``None``).
3. Load the sibling ``data/taxonomy/auto_labels_<run_id>.json``.
4. Build parallel label sequences keyed by ``(seed_group, rollout_idx)``.
5. Compute Cohen's κ via :func:`roboeval.taxonomy.agreement.cohens_kappa`.
6. Print κ + the G3 verdict (PRD §7.3 requires κ > 0.6).

Usage::

    python -m scripts.relabel_score \
        data/taxonomy/relabel_sample_18xb5ob0.json \
        data/taxonomy/relabel_sample_alr0r0p2.json

Exits 0 if every passed sample scores κ > 0.6, 1 if any falls short, 2 on
input/validation errors.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from roboeval.taxonomy.agreement import cohens_kappa

_LOG = logging.getLogger("scripts.relabel_score")

_PASS_THRESHOLD = 0.6


def _load_sample(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(
            f"{path}: expected schema_version 1, got "
            f"{payload.get('schema_version')!r}"
        )
    return payload


def _check_unlock(sample: dict[str, object], *, now: _dt.datetime) -> None:
    raw = sample.get("unlock_at")
    if not isinstance(raw, str):
        raise ValueError("sample is missing `unlock_at` (str)")
    unlock = _dt.datetime.fromisoformat(raw)
    if now < unlock:
        wait = unlock - now
        raise SystemExit(
            f"Embargo not yet elapsed for {sample.get('run_id')!r}. "
            f"Unlock at {unlock.isoformat()} "
            f"(in {wait.total_seconds() / 3600:.1f} h). "
            "Refusing to score before the unlock so the blinding holds. "
            "See PRD §7.3."
        )


def _check_complete(sample: dict[str, object]) -> list[dict[str, object]]:
    samples = sample.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("sample has no `samples` list")
    missing = [s for s in samples if s.get("manual_failure_mode") in (None, "")]
    if missing:
        idxs = ", ".join(
            f"({s.get('seed_group')}, {s.get('rollout_idx')})" for s in missing[:5]
        )
        raise SystemExit(
            f"{len(missing)} of {len(samples)} samples still have "
            f"`manual_failure_mode: null`. Fill them in before scoring. "
            f"Missing (first 5): {idxs}"
        )
    return samples


def _auto_labels_path(sample_path: Path, run_id: str) -> Path:
    return sample_path.parent / f"auto_labels_{run_id}.json"


def _index_auto_labels(auto_payload: dict[str, object]) -> dict[tuple[int, int], str]:
    labels = auto_payload.get("labels")
    if not isinstance(labels, list):
        raise ValueError("auto_labels payload is missing `labels` list")
    by_key: dict[tuple[int, int], str] = {}
    for entry in labels:
        key = (int(entry["seed_group"]), int(entry["rollout_idx"]))
        by_key[key] = str(entry["failure_mode"])
    return by_key


def score_one(sample_path: Path, *, now: _dt.datetime | None = None) -> float:
    """Score one relabel sample. Returns κ. Prints a per-sample summary."""
    now = now or _dt.datetime.now(_dt.UTC)
    sample = _load_sample(sample_path)
    _check_unlock(sample, now=now)
    complete = _check_complete(sample)

    run_id = str(sample["run_id"])
    auto_path = _auto_labels_path(sample_path, run_id)
    auto = json.loads(auto_path.read_text())
    auto_by_key = _index_auto_labels(auto)

    manual_seq: list[str] = []
    auto_seq: list[str] = []
    missing_auto: list[tuple[int, int]] = []
    for entry in complete:
        key = (int(entry["seed_group"]), int(entry["rollout_idx"]))
        auto_label = auto_by_key.get(key)
        if auto_label is None:
            missing_auto.append(key)
            continue
        manual_seq.append(str(entry["manual_failure_mode"]))
        auto_seq.append(auto_label)

    if missing_auto:
        raise SystemExit(
            f"{len(missing_auto)} sample entries have no matching auto label "
            f"in {auto_path.name}. First missing: {missing_auto[:3]}"
        )

    result = cohens_kappa(manual_seq, auto_seq)

    print(f"\n=== {run_id} ({sample_path.name}) ===")
    print(f"  rollouts scored        : {len(manual_seq)}")
    print(f"  observed agreement     : {result.observed_agreement:.4f}")
    print(f"  expected (chance) agree: {result.expected_agreement:.4f}")
    print(f"  Cohen's kappa          : {result.kappa:.4f}")
    if result.is_degenerate:
        print(
            "  WARN: degenerate κ (one rater used a single category) — "
            "interpret with care"
        )
    if result.kappa > _PASS_THRESHOLD:
        print(f"  verdict: PASS (κ > {_PASS_THRESHOLD})")
    else:
        print(f"  verdict: FAIL (κ ≤ {_PASS_THRESHOLD}) — investigate disagreements")

    return float(result.kappa)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Score Cohen's κ for one or more relabel samples and report "
            "whether G3 (PRD §7.3) is closed."
        )
    )
    parser.add_argument(
        "sample_paths",
        nargs="+",
        help="One or more data/taxonomy/relabel_sample_<run_id>.json paths.",
    )
    args = parser.parse_args(argv)

    kappas: list[float] = []
    for raw in args.sample_paths:
        path = Path(raw)
        if not path.exists():
            print(f"ERROR: {path} does not exist", file=sys.stderr)
            return 2
        kappas.append(score_one(path))

    all_pass = all(k > _PASS_THRESHOLD for k in kappas)
    print("\n=== Combined verdict ===")
    print(f"  samples scored: {len(kappas)}")
    print(f"  per-sample κ  : {[round(k, 4) for k in kappas]}")
    if all_pass:
        print("  ALL PASS — G3 (PRD §7.3) closes.")
        return 0
    print("  NOT ALL PASS — G3 stays open.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
