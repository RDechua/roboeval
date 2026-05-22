"""Render static figures for the Phase 5 blog post.

Reads ``data/headline.json`` v2 (the self-contained dashboard data
artifact) and emits PNG figures into ``docs/figures/``. The figures
mirror panels of the live dashboard but are baked at commit time so the
blog post stays self-contained when the dashboard Space is asleep.

Usage::

    python -m scripts.render_blog_figures
    # writes docs/figures/cross_axis_degradation.png
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

_LOG = logging.getLogger("scripts.render_blog_figures")

_PRIMARY_COLOR = "#2E86AB"
_RIBBON_COLOR = (46 / 255, 134 / 255, 171 / 255, 0.18)


def _load_cells(headline_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(headline_path.read_text())
    if payload.get("schema_version") != 2:
        raise ValueError(
            f"render_blog_figures needs headline.json schema 2, "
            f"got {payload.get('schema_version')!r}"
        )
    cells = payload["cells"]
    if not isinstance(cells, list):
        raise ValueError("headline.json `cells` is not a list")
    return cells


def render_cross_axis_degradation(*, headline_path: Path, out_path: Path) -> None:
    """Render the side-by-side spatial + temporal degradation panels.

    Args:
        headline_path: Path to ``data/headline.json`` (schema v2).
        out_path: PNG output path. Parent directory is created if absent.
    """
    cells = _load_cells(headline_path)

    spatial = sorted(
        [c for c in cells if c["axis"] in ("spatial", "nominal")],
        key=lambda c: float(c["magnitude"]),
    )
    temporal = sorted(
        [c for c in cells if c["axis"] in ("temporal", "nominal")],
        key=lambda c: float(c["magnitude"]),
    )

    spatial_x = [float(c["magnitude"]) * 100.0 for c in spatial]  # m -> cm
    spatial_y = [float(c["mean_tsr_custom"]) for c in spatial]
    spatial_sigma = [float(c["std_tsr_custom"]) for c in spatial]

    temporal_x = [float(c["magnitude"]) for c in temporal]
    temporal_y = [float(c["mean_tsr_custom"]) for c in temporal]
    temporal_sigma = [float(c["std_tsr_custom"]) for c in temporal]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, xs, ys, sigmas, x_label in (
        (ax_l, spatial_x, spatial_y, spatial_sigma, "Cube shift (cm)"),
        (
            ax_r,
            temporal_x,
            temporal_y,
            temporal_sigma,
            "Action delay (env steps)",
        ),
    ):
        upper = [y + s for y, s in zip(ys, sigmas, strict=True)]
        lower = [y - s for y, s in zip(ys, sigmas, strict=True)]
        ax.fill_between(xs, lower, upper, color=_RIBBON_COLOR, linewidth=0)
        ax.plot(
            xs,
            ys,
            color=_PRIMARY_COLOR,
            linewidth=2.4,
            marker="o",
            markersize=6,
        )
        ax.axvline(0, color="#888", linestyle=":", linewidth=0.8)
        ax.set_xlabel(x_label)
        ax.set_ylim(0, 1.0)
        ax.grid(visible=True, alpha=0.25)

    ax_l.set_ylabel("Mean task-success rate")
    ax_l.set_title("Spatial perturbation")
    ax_r.set_title("Temporal delay")
    fig.suptitle(
        "ACT on AlohaTransferCube -- degradation across perturbation axes",
        fontsize=12,
    )
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _LOG.info("wrote %s", out_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    repo_root = Path(__file__).resolve().parents[1]
    render_cross_axis_degradation(
        headline_path=repo_root / "data" / "headline.json",
        out_path=repo_root / "docs" / "figures" / "cross_axis_degradation.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
