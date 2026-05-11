"""Week 1 smoke tests: verify the scaffolded package imports cleanly.

These tests intentionally do not exercise lerobot, gym-aloha, or torch — those
are imported lazily by :mod:`roboeval.cli` only when the ``smoke`` subcommand
is invoked. CI runs without the heavy stack installed.
"""

from __future__ import annotations

import importlib
import re

import pytest

PACKAGE_MODULES = [
    "roboeval",
    "roboeval.cli",
    "roboeval.envs",
    "roboeval.policies",
    "roboeval.evaluation",
    "roboeval.taxonomy",
    "roboeval.residual",
]


@pytest.mark.parametrize("module_name", PACKAGE_MODULES)
def test_module_importable(module_name):
    module = importlib.import_module(module_name)
    assert module is not None


def test_version_is_pep440_like():
    import roboeval

    assert re.fullmatch(
        r"\d+\.\d+\.\d+([.+-]\w+)?", roboeval.__version__
    ), f"unexpected __version__: {roboeval.__version__!r}"


def test_cli_parser_registers_smoke_command():
    from roboeval.cli import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["smoke", "--steps", "3"])
    assert args.command == "smoke"
    assert args.steps == 3
