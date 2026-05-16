"""Policy factory — dispatches on ``kind`` to a concrete loader.

The factory is the single entry point the CLI uses to construct a
:class:`~roboeval.policies.base.Policy`. v1.0 supports ``kind="act"``;
``kind="diffusion"`` is reserved for v1.1 and raises ``NotImplementedError``
with a pointer to PRD §3.2 (the non-goal that defers Diffusion Policy
until a public sim-trained checkpoint exists for ALOHA Transfer Cube).

Adding a new policy in v1.1 is a two-line change here: add a new branch
to :func:`load_policy` and a Literal value to :data:`PolicyKind`. The
rest of the harness stays untouched because the rollout loop only
depends on the :class:`Policy` Protocol.

Heavy dependencies (``lerobot``, ``torch``) are imported lazily by the
underlying ``load_*`` functions, so importing this module is cheap and
CI can statically type-check it without the full stack.
"""

from __future__ import annotations

from typing import Literal, get_args

from roboeval.policies.base import Policy

PolicyKind = Literal["act", "diffusion"]
"""Supported policy kinds. Extend here when adding a new adapter in v1.1."""

_SUPPORTED_KINDS: frozenset[str] = frozenset(get_args(PolicyKind))


def load_policy(
    kind: str,
    repo_id: str,
    *,
    task: str = "AlohaTransferCube-v0",
    device: str = "mps",
    dataset_repo_id: str | None = None,
) -> Policy:
    """Load a pretrained policy by ``kind`` and return a :class:`Policy`.

    Args:
        kind: One of :data:`PolicyKind`. ``"act"`` is the only kind
            wired up in v1.0; ``"diffusion"`` is reserved for v1.1
            (PRD §3.2 — no public sim-trained checkpoint exists for
            ALOHA Transfer Cube).
        repo_id: HuggingFace repo id of the policy checkpoint.
        task: ALOHA task id used to build the env config. Forwarded to
            the per-kind loader. Defaults to ``"AlohaTransferCube-v0"``.
        device: Preferred torch device; per-kind loaders fall back
            automatically if unavailable.
        dataset_repo_id: HuggingFace dataset id for normalisation
            statistics. Forwarded to ``load_act_policy``; ignored by
            future adapters that don't need it.

    Returns:
        A :class:`Policy`-conforming adapter, ready for the rollout loop.

    Raises:
        ValueError: If ``kind`` is not one of :data:`_SUPPORTED_KINDS`.
            The error message lists the supported kinds so the operator
            can fix the config.
        NotImplementedError: If ``kind`` is in :data:`_SUPPORTED_KINDS`
            but the adapter has not yet been implemented (``"diffusion"``
            in v1.0).
    """
    if kind not in _SUPPORTED_KINDS:
        raise ValueError(
            f"unknown policy kind {kind!r}; " f"supported: {sorted(_SUPPORTED_KINDS)}"
        )

    if kind == "act":
        from roboeval.policies.act_loader import load_act_policy

        return load_act_policy(
            repo_id=repo_id,
            task=task,
            device=device,
            dataset_repo_id=dataset_repo_id,
        )

    if kind == "diffusion":
        raise NotImplementedError(
            "Diffusion Policy adapter is deferred to v1.1 (PRD §3.2). "
            "No public sim-trained Diffusion Policy checkpoint exists "
            "for ALOHA Transfer Cube; training from scratch is out of "
            "scope for v1.0."
        )

    # Unreachable: the membership check above narrows kind to the literal
    # union, but mypy in strict mode can't verify that across the if/elif
    # ladder, so an explicit final raise keeps the function total.
    raise AssertionError(f"unhandled supported kind: {kind!r}")
