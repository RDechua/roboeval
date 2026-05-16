"""Eval-config loader with single-key ``extends:`` resolution.

The CLI's eval and calibrate subcommands previously called
``OmegaConf.load(path)`` directly. With the Week-4 perturbation suite
that pattern would force every (axis, intensity) cell to duplicate the
full ``policy:`` / ``env:`` / ``eval:`` / ``success:`` / ``wandb:``
blocks from ``act_nominal.yaml`` — error-prone and noisy when the
nominal config changes.

:func:`load_eval_config` adds the smallest possible inheritance: a top-
level ``extends:`` key pointing at another YAML, which is recursively
loaded and merged underneath the current file's keys (current wins).

Why not Hydra's ``defaults:`` composition? Hydra's full app-mode would
require restructuring how the CLI parses configs and would force every
config file into Hydra's group/structure conventions. The CLI just
takes ``--config <path>`` today; an ``extends:`` resolver preserves
that single-file-equals-single-experiment ergonomic without dragging
in the rest of Hydra.
"""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_EXTENDS_KEY: str = "extends"
"""The reserved top-level key that triggers parent loading."""

_MAX_EXTENDS_DEPTH: int = 8
"""Refuse to follow more than this many nested ``extends:`` links.

Catches cycles and prevents accidental deep chains that make the merged
config hard to reason about. Practical configs have depth 1 or 2.
"""


def load_eval_config(path: str | Path) -> DictConfig:
    """Load a YAML eval config, recursively resolving ``extends:``.

    The current file's keys override the parent's via ``OmegaConf.merge``;
    nested dict keys merge recursively, list values replace wholesale.
    The ``extends:`` key is stripped from the returned config so
    downstream code sees only the merged content.

    Args:
        path: Path to the YAML config to load. Forms with ``extends:``
            must reference paths relative to the project root (the CLI's
            working directory) or absolute paths.

    Returns:
        A merged :class:`omegaconf.DictConfig`.

    Raises:
        FileNotFoundError: If ``path`` or any ``extends:`` ancestor is
            missing.
        ValueError: If the chain exceeds :data:`_MAX_EXTENDS_DEPTH`
            (cycle or runaway nesting).
        TypeError: If the loaded YAML's top-level isn't a dict.
    """
    return _load_with_depth(Path(path), depth=0)


def _load_with_depth(path: Path, *, depth: int) -> DictConfig:
    if depth > _MAX_EXTENDS_DEPTH:
        raise ValueError(
            f"extends chain exceeded depth {_MAX_EXTENDS_DEPTH} at {path}; "
            f"check for a cycle"
        )
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")

    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise TypeError(
            f"config root must be a mapping, got {type(cfg).__name__} at {path}"
        )

    if _EXTENDS_KEY not in cfg:
        return cfg

    parent_ref = cfg.pop(_EXTENDS_KEY)
    parent_path = Path(str(parent_ref))
    parent = _load_with_depth(parent_path, depth=depth + 1)
    merged = OmegaConf.merge(parent, cfg)
    if not isinstance(merged, DictConfig):
        raise TypeError(
            f"merged config root must be a mapping, got {type(merged).__name__} "
            f"after merging {parent_path} with {path}"
        )
    return merged


__all__ = ["load_eval_config"]
