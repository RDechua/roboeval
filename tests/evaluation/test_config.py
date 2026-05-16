"""Unit tests for the eval-config loader's `extends:` resolution."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from roboeval.evaluation.config import _MAX_EXTENDS_DEPTH, load_eval_config


def _write(path: Path, body: dict) -> None:
    path.write_text(yaml.safe_dump(body))


def test_load_without_extends_returns_file_as_is(tmp_path: Path) -> None:
    p = tmp_path / "leaf.yaml"
    _write(p, {"a": 1, "b": {"c": 2}})
    cfg = load_eval_config(p)
    assert cfg.a == 1
    assert cfg.b.c == 2
    assert "extends" not in cfg


def test_extends_merges_parent_under_child(tmp_path: Path) -> None:
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    _write(
        parent,
        {"policy": {"kind": "act", "device": "mps"}, "eval": {"max_steps": 400}},
    )
    _write(child, {"extends": str(parent), "policy": {"device": "cpu"}})

    cfg = load_eval_config(child)
    assert cfg.policy.kind == "act"  # inherited from parent
    assert cfg.policy.device == "cpu"  # child override wins
    assert cfg.eval.max_steps == 400  # inherited
    assert "extends" not in cfg  # stripped from result


def test_extends_recursive_three_levels(tmp_path: Path) -> None:
    grand = tmp_path / "grand.yaml"
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    _write(grand, {"a": 1, "b": 2, "c": 3})
    _write(parent, {"extends": str(grand), "b": 20})
    _write(child, {"extends": str(parent), "c": 300})

    cfg = load_eval_config(child)
    assert cfg.a == 1  # from grand
    assert cfg.b == 20  # from parent (override of grand)
    assert cfg.c == 300  # from child (override of parent)


def test_extends_missing_parent_raises(tmp_path: Path) -> None:
    child = tmp_path / "child.yaml"
    _write(child, {"extends": str(tmp_path / "does_not_exist.yaml"), "a": 1})
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        load_eval_config(child)


def test_extends_cycle_detected_via_depth_limit(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    _write(a, {"extends": str(b), "x": 1})
    _write(b, {"extends": str(a), "y": 2})
    with pytest.raises(ValueError, match=r"depth"):
        load_eval_config(a)


def test_load_missing_file_raises_helpful_message(tmp_path: Path) -> None:
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match=r"config not found"):
        load_eval_config(missing)


def test_perturbation_block_in_child_overrides_no_parent_block(tmp_path: Path) -> None:
    """The actual Week-4 use case: child adds a perturbation block parent lacks."""
    parent = tmp_path / "parent.yaml"
    child = tmp_path / "child.yaml"
    _write(parent, {"policy": {"kind": "act"}, "eval": {"max_steps": 400}})
    _write(
        child,
        {
            "extends": str(parent),
            "perturbation": {"kind": "spatial", "dx_m": 0.0, "dy_m": 0.03},
        },
    )
    cfg = load_eval_config(child)
    assert cfg.policy.kind == "act"
    assert cfg.perturbation.kind == "spatial"
    assert cfg.perturbation.dy_m == 0.03


def test_max_depth_constant_is_at_least_four(tmp_path: Path) -> None:
    """A practical sanity check; perturbation cells are depth 1, not 7."""
    assert _MAX_EXTENDS_DEPTH >= 4
