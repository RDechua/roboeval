"""Cohen's κ inter-rater agreement for the §7.3 blinded relabel protocol.

PRD §7.3 step 4 calls for the auto-classifier to be validated against a
single-labeller blinded self-relabel via Cohen's κ, with a target of
**κ > 0.6** (substantial agreement per Landis & Koch 1977).

The implementation is the textbook two-rater κ over a discrete label
set, with one safety case: if either rater never disagrees with the
other (perfect agreement) **and** every rollout falls into the same
category, the chance-agreement denominator is zero — κ is undefined
in that degenerate case and we return 1.0 with a note (rather than
returning NaN or raising), since "every rater puts everything in the
same bucket" trivially agrees by construction.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KappaResult:
    """Cohen's κ output with the intermediate counts kept for audit.

    Attributes:
        kappa: The Cohen's κ statistic. Range [-1, 1]; > 0 is
            better-than-chance agreement; > 0.6 is "substantial" per
            Landis & Koch.
        observed_agreement: Fraction of rollouts where both raters
            agreed (also called ``p_o``).
        expected_agreement: Fraction of agreements expected by chance
            given each rater's marginal distribution (``p_e``).
        n: Number of compared label pairs.
        is_degenerate: ``True`` when ``expected_agreement == 1`` (every
            rollout in the same category); ``kappa`` is then defined
            as 1.0 by convention.
    """

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n: int
    is_degenerate: bool


def cohens_kappa(
    rater_a: Sequence[str],
    rater_b: Sequence[str],
) -> KappaResult:
    """Compute Cohen's κ between two equal-length label sequences.

    Args:
        rater_a: First rater's labels, in rollout order.
        rater_b: Second rater's labels, same rollout order, same
            length as ``rater_a``.

    Returns:
        A :class:`KappaResult` with the κ statistic and the
        intermediate counts.

    Raises:
        ValueError: If the two sequences differ in length, or if
            either is empty.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"rater_a and rater_b must have equal length; "
            f"got {len(rater_a)} vs {len(rater_b)}"
        )
    if not rater_a:
        raise ValueError("cohens_kappa requires at least one label pair")

    n = len(rater_a)
    agree = sum(1 for a, b in zip(rater_a, rater_b, strict=True) if a == b)
    p_o = agree / n

    counts_a = Counter(rater_a)
    counts_b = Counter(rater_b)
    labels = set(counts_a) | set(counts_b)

    p_e = sum((counts_a.get(c, 0) / n) * (counts_b.get(c, 0) / n) for c in labels)

    if p_e >= 1.0:
        return KappaResult(
            kappa=1.0,
            observed_agreement=p_o,
            expected_agreement=p_e,
            n=n,
            is_degenerate=True,
        )

    kappa = (p_o - p_e) / (1.0 - p_e)
    return KappaResult(
        kappa=kappa,
        observed_agreement=p_o,
        expected_agreement=p_e,
        n=n,
        is_degenerate=False,
    )


__all__ = ["KappaResult", "cohens_kappa"]
