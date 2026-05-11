"""Smoke test for the ALOHA env factory.

Marked ``slow`` because constructing the env imports MuJoCo + dm_control and
takes ~1 second; CI skips it via ``-m "not slow"``.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.slow
def test_make_aloha_env_resets_and_steps():
    from roboeval.envs.aloha import (
        ALOHA_TRANSFER_CUBE_ID,
        get_cube_state,
        make_aloha_env,
    )

    assert ALOHA_TRANSFER_CUBE_ID == "gym_aloha/AlohaTransferCube-v0"
    env = make_aloha_env()
    try:
        obs, info = env.reset(seed=0)
        assert "agent_pos" in obs
        assert "pixels" in obs
        cube = get_cube_state(env)
        assert cube.shape == (7,)
        assert cube.dtype == np.float64
        # Take one random step to confirm the action space works
        action = env.action_space.sample()
        obs, _r, _term, _trunc, _info = env.step(action)
        cube_after = get_cube_state(env)
        assert cube_after.shape == (7,)
    finally:
        env.close()
