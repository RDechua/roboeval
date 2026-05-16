"""Unit tests for the policy factory dispatch (no lerobot import)."""

from __future__ import annotations

import pytest

from roboeval.policies.factory import load_policy


def test_unknown_kind_raises_with_helpful_message():
    with pytest.raises(ValueError, match=r"unknown policy kind 'not_a_kind'"):
        load_policy(kind="not_a_kind", repo_id="anywhere")


def test_unknown_kind_lists_supported_kinds():
    with pytest.raises(ValueError, match=r"supported: \[.*'act'.*'diffusion'.*\]"):
        load_policy(kind="xyz", repo_id="anywhere")


def test_diffusion_kind_raises_not_implemented_with_prd_pointer():
    # Diffusion is reserved in the Literal but no adapter exists yet (v1.1).
    with pytest.raises(NotImplementedError, match=r"v1\.1.*PRD"):
        load_policy(kind="diffusion", repo_id="lerobot/diffusion_pusht")


def test_empty_kind_rejected():
    with pytest.raises(ValueError, match="unknown policy kind"):
        load_policy(kind="", repo_id="lerobot/act_x")
