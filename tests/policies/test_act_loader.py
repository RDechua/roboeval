"""Smoke test for the ACT checkpoint loader.

Marked ``slow`` because it downloads the ~80 MB ACT checkpoint from the
HuggingFace Hub on first run. CI skips slow tests; run locally with
``pytest -m slow tests/policies``.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_load_act_policy_returns_adapter():
    from roboeval.policies.act_loader import ACTPolicyAdapter, load_act_policy
    from roboeval.policies.base import Policy

    adapter = load_act_policy(
        repo_id="lerobot/act_aloha_sim_transfer_cube_human",
        task="AlohaTransferCube-v0",
        device="cpu",
    )
    assert isinstance(adapter, ACTPolicyAdapter)
    # runtime_checkable Protocol — confirms the adapter satisfies Policy
    assert isinstance(adapter, Policy)
    assert adapter.policy_id == "lerobot/act_aloha_sim_transfer_cube_human"
    assert adapter.device in {"cpu", "mps", "cuda"}


@pytest.mark.slow
def test_act_policy_produces_action_of_correct_shape():
    from roboeval.envs.aloha import make_aloha_env
    from roboeval.policies.act_loader import load_act_policy

    env = make_aloha_env()
    adapter = load_act_policy(
        repo_id="lerobot/act_aloha_sim_transfer_cube_human",
        task="AlohaTransferCube-v0",
        device="cpu",
    )
    obs, _info = env.reset(seed=0)
    adapter.reset()
    action = adapter.select_action(obs)
    assert action.shape == (14,)
    assert action.dtype == np.float32
    env.close()
