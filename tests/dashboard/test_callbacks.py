"""Smoke tests for the Dash app skeleton at analysis/dashboard/app.py."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

dash = pytest.importorskip("dash")

_APP_PATH = Path(__file__).resolve().parents[2] / "analysis" / "dashboard"


def _load_app_module() -> ModuleType:
    if str(_APP_PATH) not in sys.path:
        sys.path.insert(0, str(_APP_PATH))
    if "app" in sys.modules:
        return importlib.reload(sys.modules["app"])
    return importlib.import_module("app")


def test_app_module_imports_and_exposes_server() -> None:
    mod = _load_app_module()
    assert hasattr(mod, "server")
    assert hasattr(mod, "app")


def test_app_layout_has_hero_curve_and_failure_stack() -> None:
    mod = _load_app_module()
    layout_html = str(mod.app.layout)
    assert "hero-curve" in layout_html
    assert "failure-stack" in layout_html
    assert "ablation-plot" in layout_html


def test_update_hero_callback_round_trip() -> None:
    mod = _load_app_module()
    fig_dict = mod.update_hero(
        axis="spatial",
        metric="mean_tsr_custom",
        data=mod._STORE_PAYLOAD,
    )
    assert "data" in fig_dict
    assert "layout" in fig_dict


def test_update_failure_stack_callback_changes_with_cell() -> None:
    mod = _load_app_module()
    a = mod.update_failure_stack(cell_id="y+5cm", data=mod._STORE_PAYLOAD)
    b = mod.update_failure_stack(cell_id="y-5cm", data=mod._STORE_PAYLOAD)
    assert a["layout"]["title"]["text"] != b["layout"]["title"]["text"]
