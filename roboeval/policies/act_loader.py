"""ACT checkpoint loader implementing :class:`Policy`.

Builds a LeRobot ``ACTPolicy`` from a HuggingFace repo id and wraps it in an
adapter that takes raw gymnasium observation dicts (numpy) and returns numpy
actions. All torch / device / normalisation handling is internal — the
rollout loop only sees the :class:`Policy` Protocol.

Heavy dependencies (``lerobot``, ``torch``, ``mujoco``) are imported lazily
inside :func:`load_act_policy` so that importing this module is cheap and
CI can statically type-check it without the full stack.

The dataclass fields hold lerobot internals that lack public type stubs;
we mark those ``Any`` and route the file through an ``ANN401`` per-file
ruff ignore (see ``pyproject.toml``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch

from roboeval.policies.base import ObservationDict


def _detect_device(prefer: str) -> str:
    """Pick a torch device, falling back if ``prefer`` is unavailable.

    Args:
        prefer: One of ``"mps"``, ``"cuda"``, ``"cpu"``.

    Returns:
        A device string usable in ``torch.device(...)``. Falls back to
        ``"cpu"`` if neither MPS nor CUDA is available.
    """
    if prefer == "mps" and torch.backends.mps.is_available():
        return "mps"
    if prefer == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _add_batch_dim(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap every leaf ``np.ndarray`` in a leading length-1 batch dim.

    LeRobot's ``preprocess_observation`` expects vector-env shapes
    ``(n_envs, ...)``; we run a single env, so we have to add the batch
    dim manually before calling it. Recursive to handle nested obs dicts
    such as ALOHA's ``{"pixels": {"top": ...}}``.

    Args:
        obs: Raw gym observation dict (single env, no batch dim).

    Returns:
        Same structure, every array prefixed with a length-1 batch axis.
    """
    out: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, Mapping):
            out[key] = _add_batch_dim(value)
        elif isinstance(value, np.ndarray):
            out[key] = value[None, ...]
        else:
            out[key] = value
    return out


@dataclass
class ACTPolicyAdapter:
    """Adapter wrapping LeRobot's ``ACTPolicy`` behind the :class:`Policy` protocol.

    Construct via :func:`load_act_policy`; not intended to be instantiated
    directly. The non-private attributes are part of the public surface for
    logging and tests (``policy_id``, ``device``). Everything else is
    lerobot internals.

    Attributes:
        policy_id: HuggingFace repo id the checkpoint was loaded from.
        device: Torch device string the policy lives on.
    """

    policy_id: str
    device: str
    _policy: Any = field(repr=False)
    _preprocessor: Any = field(repr=False)
    _postprocessor: Any = field(repr=False)
    _env_preprocessor: Any = field(repr=False)
    _env_postprocessor: Any = field(repr=False)

    def reset(self) -> None:
        """Clear ACT's internal action-chunk queue. Call before each rollout."""
        self._policy.reset()

    def select_action(self, observation: ObservationDict) -> npt.NDArray[np.float32]:
        """Compute the next action for a single ALOHA pixels_agent_pos obs.

        Pipeline:
            1. Add a length-1 batch dim to every leaf array
            2. ``preprocess_observation`` (LeRobot util) — converts env-side
               keys (``pixels/top``, ``agent_pos``) to policy-side keys
               (``observation.images.top``, ``observation.state``) and to
               torch tensors
            3. Apply env-level + policy-level preprocessors (normalisation)
            4. ``policy.select_action`` (chunked inference) under
               ``torch.inference_mode()``
            5. Apply policy + env postprocessors (action denormalisation)
            6. Strip batch dim, move to CPU, cast to ``float32``

        Args:
            observation: Single-env gym observation dict.

        Returns:
            1-D ``float32`` action array of length 14, in the env's
            joint-position action space.
        """
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION

        batched = _add_batch_dim(observation)
        batch = preprocess_observation(batched)
        batch = self._env_preprocessor(batch)
        batch = self._preprocessor(batch)

        with torch.inference_mode():
            action_tensor = self._policy.select_action(batch)

        action_tensor = self._postprocessor(action_tensor)
        action_transition = {ACTION: action_tensor}
        action_transition = self._env_postprocessor(action_transition)
        action_tensor = action_transition[ACTION]

        # ACT returns shape (batch=1, action_dim=14); strip batch axis.
        action_np = action_tensor.detach().to("cpu").numpy().astype(np.float32)
        return cast(npt.NDArray[np.float32], action_np[0])


def load_act_policy(
    repo_id: str,
    task: str = "AlohaTransferCube-v0",
    device: str = "mps",
    dataset_repo_id: str | None = None,
) -> ACTPolicyAdapter:
    """Load an ACT checkpoint from HuggingFace as a :class:`Policy`.

    The v0.1-era ACT checkpoints on the LeRobot org (incl.
    ``lerobot/act_aloha_sim_transfer_cube_human``, May 2026 audit) ship
    ``config.json`` + ``model.safetensors`` but **not** the
    ``policy_preprocessor.json`` / ``policy_postprocessor.json`` files
    introduced by LeRobot 0.4.x. We therefore reconstruct the
    normalisation processors from the source dataset's statistics via
    :class:`LeRobotDatasetMetadata`, and pass the same stats into
    ``make_policy`` so the policy's normalisation buffers are populated
    before weight loading.

    Args:
        repo_id: HuggingFace repo id of the policy, e.g.
            ``"lerobot/act_aloha_sim_transfer_cube_human"``.
        task: ALOHA task id used to build the env config. Defaults to
            ``"AlohaTransferCube-v0"``.
        device: Preferred torch device; falls back to CPU automatically
            via :func:`_detect_device`.
        dataset_repo_id: HuggingFace dataset id used to pull normalisation
            statistics. When ``None``, defaults to the policy repo's name
            with a leading ``"act_"`` removed (e.g. the policy
            ``lerobot/act_aloha_sim_transfer_cube_human`` maps to the
            dataset ``lerobot/aloha_sim_transfer_cube_human``).

    Returns:
        An :class:`ACTPolicyAdapter` ready to feed into the rollout loop.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.envs.configs import AlohaEnv as LRAlohaCfg
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import (
        make_policy,
        make_policy_config,
        make_pre_post_processors,
    )

    if dataset_repo_id is None:
        owner, name = repo_id.split("/", 1)
        dataset_repo_id = f"{owner}/{name.removeprefix('act_')}"

    resolved = _detect_device(device)

    ds_meta = LeRobotDatasetMetadata(repo_id=dataset_repo_id)
    env_cfg = LRAlohaCfg(task=task)

    policy_cfg = make_policy_config("act")
    policy_cfg.pretrained_path = repo_id
    policy_cfg.device = resolved

    # ds_meta path: features inferred from the dataset, normalisation
    # stats also pulled in so they are applied during weight loading.
    policy = make_policy(policy_cfg, ds_meta=ds_meta)

    # No pretrained_path here — falls through to the "create new processors
    # from dataset stats" branch of make_pre_post_processors.
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg, dataset_stats=ds_meta.stats
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy_cfg
    )

    return ACTPolicyAdapter(
        policy_id=repo_id,
        device=resolved,
        _policy=policy,
        _preprocessor=preprocessor,
        _postprocessor=postprocessor,
        _env_preprocessor=env_preprocessor,
        _env_postprocessor=env_postprocessor,
    )
