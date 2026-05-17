r"""TSR-vs-perturbation degradation curve with Bernoulli SE floor overlay.

Renders the second panel planned for the PRD §6.4 figure: mean TSR
against perturbation magnitude on the x-axis, with seed-group sigma error
bars and a dashed Bernoulli-SE floor line ``√(p(1-p)/N)``. The floor
visualises which cells are over-dispersed (sigma > floor — seed-to-seed
variance unexplained by binomial sampling alone) vs under-dispersed
(sigma < floor — competence-collapse signature where the policy fails
deterministically).

Inputs are the same ``auto_labels_<run_id>.json`` files produced by
``roboeval evaluate`` (or post-hoc by ``relabel_from_wandb.py``). Per-
seed-group TSR is recomputed from the ``labels`` array; the script
doesn't trust the ``distribution`` block since aggregate-only files
can't supply sigma.

Usage
-----
::

    python scripts/plot_degradation_curve.py \\
        --cell=-5:-5.0:data/taxonomy/auto_labels_18xb5ob0.json \\
        --cell=-3:-3.0:data/taxonomy/auto_labels_11ugk2a3.json \\
        ...
        --cell=+5:5.0:data/taxonomy/auto_labels_alr0r0p2.json \\
        --out docs/figures/spatial_degradation_curve.png

Each ``--cell`` is ``<label>:<x_value>:<json_path>``. ``x_value`` is
a float (cm, degrees, steps, etc.) used as the x-axis coordinate.
``label`` is the bar/point label.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Any


def per_seed_group_tsr(labels_obj: dict[str, Any]) -> tuple[list[float], int]:
    """Recompute per-seed-group TSR from the ``labels`` array.

    Args:
        labels_obj: Parsed auto_labels JSON dict.

    Returns:
        ``(per_seed_tsr_list, n_total)`` where the list has one entry
        per seed group (sorted by seed_group id).
    """
    by_seed: dict[int, list[bool]] = defaultdict(list)
    for label in labels_obj.get("labels", []):
        # success := failure_mode is None (per io.py contract).
        success = label.get("failure_mode") is None
        by_seed[int(label.get("seed_group", 0))].append(success)
    if not by_seed:
        return [], 0
    per_seed_tsr: list[float] = []
    n_total = 0
    for _sg, rollouts in sorted(by_seed.items()):
        per_seed_tsr.append(sum(rollouts) / len(rollouts))
        n_total += len(rollouts)
    return per_seed_tsr, n_total


def aggregate_cell(json_path: Path) -> tuple[float, float, int]:
    """Return ``(mean_tsr, sigma, n)`` for one auto_labels JSON.

    sigma is the population stdev across seed groups (matches the PRD §6.3
    reporting convention). For single-seed-group files sigma defaults to 0.
    """
    obj = json.loads(json_path.read_text())
    per_seed_tsr, n_total = per_seed_group_tsr(obj)
    if not per_seed_tsr:
        return 0.0, 0.0, 0
    mean = statistics.fmean(per_seed_tsr)
    sigma = statistics.pstdev(per_seed_tsr) if len(per_seed_tsr) > 1 else 0.0
    return mean, sigma, n_total


def bernoulli_se(p: float, n: int) -> float:
    """Bernoulli standard error √(p(1-p)/n). Zero if n is non-positive."""
    if n <= 0:
        return 0.0
    return sqrt(p * (1.0 - p) / n)


def render_curve(
    cells: list[tuple[str, float, float, float, int]],
    out_path: Path,
) -> Path:
    """Render the TSR-vs-x degradation panel.

    Args:
        cells: ``[(label, x_value, mean_tsr, sigma, n), ...]`` ordered
            by ``x_value``.
        out_path: Output PNG path. Parents created if missing.

    Returns:
        The same ``out_path`` for caller chaining.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells_sorted = sorted(cells, key=lambda c: c[1])
    labels = [c[0] for c in cells_sorted]
    xs = [c[1] for c in cells_sorted]
    means = [c[2] for c in cells_sorted]
    sigmas = [c[3] for c in cells_sorted]
    ns = [c[4] for c in cells_sorted]
    floors = [bernoulli_se(p, n) for p, n in zip(means, ns, strict=True)]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    ax.errorbar(
        xs,
        means,
        yerr=sigmas,
        fmt="o-",
        capsize=4,
        color="#1f77b4",
        ecolor="#1f77b4",
        label="mean TSR  ±sigma (across 3 seed groups)",
    )
    ax.plot(
        xs,
        [m + f for m, f in zip(means, floors, strict=True)],
        linestyle="--",
        color="#7f7f7f",
        alpha=0.7,
        label="mean + Bernoulli SE  √(p(1-p)/N)",
    )
    ax.plot(
        xs,
        [m - f for m, f in zip(means, floors, strict=True)],
        linestyle="--",
        color="#7f7f7f",
        alpha=0.7,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean TSR")
    ax.set_xlabel("Perturbation magnitude")
    ax.set_title(
        "ACT degradation curve with Bernoulli SE floor overlay\n"
        "(sigma below the floor signals competence collapse)"
    )
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")
    ax.legend(loc="lower center", frameon=False, fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_cell(arg: str) -> tuple[str, float, Path]:
    """Parse a ``<label>:<x_value>:<json_path>`` CLI arg."""
    parts = arg.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"--cell expects '<label>:<x>:<path>'; got {arg!r}"
        )
    label, x_str, raw_path = parts
    try:
        x_value = float(x_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--cell x_value must be float; got {x_str!r}"
        ) from exc
    return label.strip(), x_value, Path(raw_path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        type=_parse_cell,
        help="One '<label>:<x_value>:<json_path>' per perturbation cell.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/figures/spatial_degradation_curve.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    cells: list[tuple[str, float, float, float, int]] = []
    for label, x_value, json_path in args.cell:
        mean, sigma, n = aggregate_cell(json_path)
        cells.append((label, x_value, mean, sigma, n))
    out_path = render_curve(cells, args.out)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
