r"""Generate the spatial-axis failure-distribution figure for PRD §6.4.

Reads one ``data/taxonomy/auto_labels_<run_id>.json`` per perturbation
cell and produces a stacked-bar figure showing the fraction of rollouts
in each failure category. The "success" bucket is drawn at the bottom
so the figure visually splits into a "successes (top of bar)" vs.
"failures (rest)" reading at a glance.

Usage
-----
::

    python scripts/plot_failure_distribution.py \\
        --cell "nominal:data/taxonomy/auto_labels_cm6uf89g.json" \\
        --cell "+1cm:data/taxonomy/auto_labels_p2pltgd8.json" \\
        --cell "+3cm:data/taxonomy/auto_labels_miuy4kux.json" \\
        --cell "+5cm:data/taxonomy/auto_labels_alr0r0p2.json" \\
        --out docs/figures/spatial_failure_distribution.png

Each ``--cell`` argument is ``<label>:<json_path>``. Order on the
command line determines x-axis order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Order matters for the stack — successes at the bottom, then specific
# failure modes ordered roughly by "specificity": grasp/approach are
# outcome-level, recovery is a motor pattern, oscillation is a motor
# pathology, timeout is generic, needs_review is the fallthrough.
_BUCKET_ORDER: list[str] = [
    "success",
    "grasp_failure",
    "approach_failure",
    "recovery_failure",
    "action_oscillation",
    "timeout",
    "visual_confusion",
    "needs_review",
]

_BUCKET_COLORS: dict[str, str] = {
    "success": "#2ca02c",  # green
    "grasp_failure": "#d62728",  # red
    "approach_failure": "#ff7f0e",  # orange
    "recovery_failure": "#1f77b4",  # blue
    "action_oscillation": "#9467bd",  # purple
    "timeout": "#8c564b",  # brown
    "visual_confusion": "#e377c2",  # pink
    "needs_review": "#7f7f7f",  # grey
}

_BUCKET_LABELS: dict[str, str] = {
    "success": "Success",
    "grasp_failure": "Grasp",
    "approach_failure": "Approach",
    "recovery_failure": "Recovery",
    "action_oscillation": "Oscillation",
    "timeout": "Timeout",
    "visual_confusion": "Visual",
    "needs_review": "Needs review",
}


def load_distribution(json_path: Path) -> tuple[dict[str, int], int]:
    """Load one auto-labels JSON and return ``(distribution, n_rollouts)``."""
    obj = json.loads(json_path.read_text())
    return dict(obj["distribution"]), int(obj["n_rollouts"])


def build_stack_arrays(
    cells: list[tuple[str, dict[str, int], int]],
) -> tuple[list[str], dict[str, list[float]]]:
    """Convert (label, distribution, n) tuples into stack-ready percentages.

    Returns the x-axis labels and a dict keyed by bucket name mapping to
    a list of per-cell percentages (length == len(cells)).
    """
    x_labels = [label for label, _, _ in cells]
    stacks: dict[str, list[float]] = {b: [] for b in _BUCKET_ORDER}
    for _label, dist, n in cells:
        for bucket in _BUCKET_ORDER:
            count = int(dist.get(bucket, 0))
            pct = (count / n * 100.0) if n > 0 else 0.0
            stacks[bucket].append(pct)
    return x_labels, stacks


def render_figure(
    cells: list[tuple[str, dict[str, int], int]],
    out_path: Path,
) -> Path:
    """Render the stacked-bar PNG to ``out_path``. Returns the same path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_labels, stacks = build_stack_arrays(cells)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    bottom = [0.0] * len(x_labels)
    for bucket in _BUCKET_ORDER:
        values = stacks[bucket]
        if all(v == 0.0 for v in values):
            # Skip zero-everywhere buckets to keep the legend uncluttered.
            continue
        ax.bar(
            x_labels,
            values,
            bottom=bottom,
            label=_BUCKET_LABELS[bucket],
            color=_BUCKET_COLORS[bucket],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = [b + v for b, v in zip(bottom, values, strict=True)]
    ax.set_ylim(0, 100)
    ax.set_ylabel("Fraction of rollouts (%)")
    ax.set_xlabel("Spatial y-shift")
    ax.set_title(
        "ACT failure-mode distribution across spatial perturbation\n"
        "(3 seed groups, 50 rollouts per cell)"
    )
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _parse_cell(arg: str) -> tuple[str, Path]:
    """Parse a ``<label>:<json_path>`` CLI arg."""
    if ":" not in arg:
        raise argparse.ArgumentTypeError(
            f"--cell expects '<label>:<path>'; got {arg!r}"
        )
    label, _, raw_path = arg.partition(":")
    return label.strip(), Path(raw_path.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        type=_parse_cell,
        help="One '<label>:<json_path>' pair per perturbation cell. "
        "Order determines x-axis order.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/figures/spatial_failure_distribution.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    cells: list[tuple[str, dict[str, int], int]] = []
    for label, json_path in args.cell:
        distribution, n = load_distribution(json_path)
        cells.append((label, distribution, n))
    out_path = render_figure(cells, args.out)
    print(f"Wrote {out_path}")
    return 0


_TableRow = tuple[str, int, int, int, int, int, int, int, int]


def emit_markdown_table(
    cells: list[tuple[str, dict[str, int], int]],
) -> str:
    """Render the same data as a Markdown table for the research log."""
    header_cols = ["cell", "n"] + [_BUCKET_LABELS[b] for b in _BUCKET_ORDER]
    rows: list[list[str]] = [header_cols, ["---"] * len(header_cols)]
    for label, dist, n in cells:
        row: list[str] = [label, str(n)]
        for bucket in _BUCKET_ORDER:
            count = int(dist.get(bucket, 0))
            pct = (count / n * 100.0) if n > 0 else 0.0
            row.append(f"{count} ({pct:.1f}%)")
        rows.append(row)
    return "\n".join("| " + " | ".join(r) + " |" for r in rows)


if __name__ == "__main__":
    sys.exit(main())
