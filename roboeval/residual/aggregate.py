"""Aggregate Phase 4 ablation results (PRD §8.3).

Reads N ``eval_results_<run_id>.json`` artifacts (produced by
:mod:`roboeval.evaluation.results_io`), groups them by condition
(A — frozen base, B — sparse residual, C — shaped residual), and
computes the PRD §8.3 reporting stack:

* mean ± std of ``mean_tsr_custom`` across seed groups per condition,
* percentile bootstrap CI on the mean (10 000 resamples, seed=0),
* delta-TSR vs Condition A,
* one-sided Welch's t-test (alpha = 0.05) for each non-baseline condition.

The stats math is pure stdlib so the aggregator runs in CI without
``scipy`` / ``numpy``. The Student-t survival function uses the
regularised incomplete beta computed via the Lentz continued-fraction
expansion (Numerical Recipes section 6.4) -- accurate at the low
degrees of freedom (df approx 2-4) that 3-seed comparisons produce,
where a normal approximation would mis-report significance.

Public surface:

* :func:`classify_condition` — payload → ``"A"`` / ``"B"`` / ``"C"`` /
  ``"unknown"``.
* :func:`load_eval_results` — read all matching JSONs from a directory.
* :func:`aggregate_runs` — build an :class:`AblationReport`.
* :func:`report_to_dict` / :func:`format_markdown` — serialisers.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

SCHEMA_VERSION = 1

BOOTSTRAP_DEFAULT_RESAMPLES = 10_000
BOOTSTRAP_DEFAULT_SEED = 0
SIGNIFICANCE_ALPHA = 0.05

CONDITION_LABELS: dict[str, str] = {
    "A": "Frozen base only",
    "B": "Residual RL, sparse reward",
    "C": "Residual RL, shaped reward",
}


# --------------------------------------------------------------------------- #
# Dataclasses                                                                 #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConditionStats:
    """Per-condition aggregate metrics.

    All TSR values are on the secondary (geometric) success signal
    (``mean_tsr_custom``); the primary gym-aloha signal is unreliable
    under physical perturbation (PRD §6.2).

    Attributes:
        condition_id: ``"A"`` / ``"B"`` / ``"C"``.
        label: Human-readable PRD §8.3 condition label.
        n_runs: Number of evaluation runs (typically 3 — one per seed).
        per_seed_means: One ``mean_tsr_custom`` per evaluation run (each
            run already aggregates across its own seed groups).
        mean: Mean of ``per_seed_means``.
        std: Population stdev of ``per_seed_means`` (matches PRD
            "mean ± std" reporting convention).
        bootstrap_ci_low: 2.5th percentile of bootstrap distribution.
        bootstrap_ci_high: 97.5th percentile of bootstrap distribution.
        n_rollouts: Total rollouts across all runs (for context only).
        run_ids: ``run_id`` of each contributing run.
    """

    condition_id: str
    label: str
    n_runs: int
    per_seed_means: tuple[float, ...]
    mean: float
    std: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    n_rollouts: int
    run_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ConditionComparison:
    """Pairwise statistical comparison of one condition against A (baseline).

    Welch's t-test (unequal variances), one-sided ``H1: μ_test > μ_A``.
    With n=3 seeds per condition the test has limited power; the PRD
    explicitly accepts honest null results as a deliverable.

    Attributes:
        condition_id: The non-baseline condition being compared
            (``"B"`` or ``"C"``).
        delta_tsr: ``mean(test) - mean(A)``.
        t_statistic: Welch's t-statistic.
        df: Welch-Satterthwaite degrees of freedom.
        p_value: One-sided p-value (``H1: μ_test > μ_A``). ``None`` when
            the test cannot be computed (e.g. zero variance + identical
            means, or insufficient samples).
        significant_at_05: ``p_value < 0.05``; ``False`` when
            ``p_value`` is ``None``.
    """

    condition_id: str
    delta_tsr: float
    t_statistic: float | None
    df: float | None
    p_value: float | None
    significant_at_05: bool


@dataclass(frozen=True, slots=True)
class AblationReport:
    """Top-level Phase 4 ablation report (PRD §8.3 deliverable).

    Attributes:
        schema_version: Bumped on any backward-incompatible change.
        target_perturbation: ``{"kind": ..., **params}`` describing the
            common cell the ablation targets (e.g. ``+5cm``).
        conditions: Per-condition aggregates, ordered ``A`` → ``B`` → ``C``.
            Conditions with no contributing runs are omitted.
        comparisons: One per non-baseline condition present in
            ``conditions``; empty when Condition A is absent.
        warnings: Free-form strings surfacing data-quality issues
            (missing condition, single-run condition, schema mismatch).
    """

    schema_version: int
    target_perturbation: dict[str, Any]
    conditions: tuple[ConditionStats, ...]
    comparisons: tuple[ConditionComparison, ...]
    warnings: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Classification                                                              #
# --------------------------------------------------------------------------- #


def classify_condition(payload: dict[str, Any]) -> str:
    """Return ``"A"`` / ``"B"`` / ``"C"`` / ``"unknown"`` for one payload.

    * ``A`` — base ACT policy (``policy_kind == "act"``), no residual block.
    * ``B`` — residual policy with ``residual.reward_kind == "sparse"``.
    * ``C`` — residual policy with ``residual.reward_kind == "shaped"``.

    Any payload that doesn't match these patterns is classified
    ``"unknown"`` and (when surfaced via :func:`load_eval_results`) ends
    up in a warning rather than crashing the aggregator — a half-typed
    schema shouldn't sink the whole report.
    """
    policy_kind = payload.get("policy_kind")
    residual = payload.get("residual")
    if policy_kind == "act" and residual is None:
        return "A"
    if policy_kind == "residual_act" and isinstance(residual, dict):
        reward_kind = residual.get("reward_kind")
        if reward_kind == "sparse":
            return "B"
        if reward_kind == "shaped":
            return "C"
    return "unknown"


def _perturbation_signature(
    payload: dict[str, Any],
) -> tuple[str, tuple[tuple[str, float], ...]]:
    """Hashable signature of the perturbation cell a run targets.

    Two runs targeting the same cell share this signature. Used to
    detect mixed-cell ablations (which the aggregator refuses to
    compare).
    """
    kind = str(payload.get("perturbation_kind", "none"))
    params = payload.get("perturbation_params", {}) or {}
    items = tuple(sorted((str(k), float(v)) for k, v in params.items()))
    return (kind, items)


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #


def load_eval_results(paths: Iterable[Path | str]) -> list[dict[str, Any]]:
    """Load and JSON-decode each path. Files are returned in input order.

    Raises:
        FileNotFoundError: A path doesn't exist.
        json.JSONDecodeError: A file isn't valid JSON.
    """
    payloads: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        payloads.append(json.loads(path.read_text()))
    return payloads


# --------------------------------------------------------------------------- #
# Stats: Welch's t-test + bootstrap CI (stdlib only)                          #
# --------------------------------------------------------------------------- #


def _beta_continued_fraction(x: float, a: float, b: float) -> float:
    """Lentz continued-fraction expansion for the incomplete beta (NR §6.4)."""
    max_iter = 200
    eps = 1e-12
    tiny = 1e-30

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        # Even step
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        # Odd step
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    return h


def _regularised_incomplete_beta(x: float, a: float, b: float) -> float:
    """``I_x(a, b)``: regularised incomplete beta function.

    Stable across the full domain via the standard transform
    ``I_x(a, b) = 1 - I_{1-x}(b, a)`` when ``x`` is large.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _beta_continued_fraction(x, a, b) / a
    return 1.0 - bt * _beta_continued_fraction(1.0 - x, b, a) / b


def student_t_sf(t: float, df: float) -> float:
    """``P(T > t)`` for Student's t with ``df`` degrees of freedom.

    Returns 0.5 when ``t == 0``. ``t < 0`` uses the symmetry
    ``sf(t) = 1 - sf(-t)``. Accurate at low df where normal
    approximations diverge.
    """
    if df <= 0:
        raise ValueError(f"df must be positive, got {df!r}")
    if t == 0.0:
        return 0.5
    if math.isinf(t):
        return 0.0 if t > 0 else 1.0
    if t < 0:
        return 1.0 - student_t_sf(-t, df)
    x = df / (df + t * t)
    return 0.5 * _regularised_incomplete_beta(x, df / 2.0, 0.5)


def welch_t_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
) -> tuple[float | None, float | None, float | None]:
    """One-sided Welch's t-test, ``H1: mean(b) > mean(a)``.

    Returns:
        ``(t_statistic, df, p_value)``. All three are ``None`` when the
        test cannot be computed: too few samples (n < 2 in either arm),
        or zero variance in both arms with non-zero mean difference.
    """
    n_a, n_b = len(values_a), len(values_b)
    if n_a < 2 or n_b < 2:
        return (None, None, None)

    mean_a = fmean(values_a)
    mean_b = fmean(values_b)
    # Sample variance (n-1 denominator).
    var_a = sum((x - mean_a) ** 2 for x in values_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in values_b) / (n_b - 1)

    if var_a == 0.0 and var_b == 0.0:
        if mean_a == mean_b:
            return (0.0, float(n_a + n_b - 2), 0.5)
        return (None, None, None)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0.0:
        return (None, None, None)
    t = (mean_b - mean_a) / se

    # Welch-Satterthwaite degrees of freedom.
    num = (var_a / n_a + var_b / n_b) ** 2
    den_a = (var_a / n_a) ** 2 / (n_a - 1) if n_a > 1 else 0.0
    den_b = (var_b / n_b) ** 2 / (n_b - 1) if n_b > 1 else 0.0
    den = den_a + den_b
    if den == 0.0:
        return (None, None, None)
    df = num / den

    p = student_t_sf(t, df)
    return (t, df, p)


def bootstrap_ci_mean(
    values: Sequence[float],
    *,
    n_resamples: int = BOOTSTRAP_DEFAULT_RESAMPLES,
    confidence: float = 0.95,
    rng_seed: int = BOOTSTRAP_DEFAULT_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean.

    With n=3 seed groups the bootstrap distribution is coarse (only 27
    distinct multisets) — caller is responsible for not over-interpreting.
    Use a deterministic ``rng_seed`` so repeated reports are bit-identical.

    Returns:
        ``(low, high)`` for the central ``confidence`` interval.

    Raises:
        ValueError: ``values`` is empty.
    """
    if not values:
        raise ValueError("bootstrap_ci_mean requires at least one value")
    rng = random.Random(rng_seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = max(0, int(math.floor(alpha * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1.0 - alpha) * n_resamples)) - 1)
    return (means[lo_idx], means[hi_idx])


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def _condition_from_payload(payload: dict[str, Any]) -> tuple[str, float, int, str]:
    """Extract ``(condition_id, mean_tsr_custom, n_rollouts, run_id)``.

    Pulled out so :func:`aggregate_runs` and tests don't both
    have to know the schema keys.
    """
    cid = classify_condition(payload)
    metrics = payload.get("metrics", {})
    mean_tsr_custom = float(metrics.get("mean_tsr_custom", 0.0))
    n_rollouts = int(metrics.get("n_rollouts", 0))
    run_id = str(payload.get("run_id", ""))
    return (cid, mean_tsr_custom, n_rollouts, run_id)


def aggregate_runs(
    payloads: Sequence[dict[str, Any]],
    *,
    bootstrap_resamples: int = BOOTSTRAP_DEFAULT_RESAMPLES,
    bootstrap_seed: int = BOOTSTRAP_DEFAULT_SEED,
) -> AblationReport:
    """Build an :class:`AblationReport` from N eval-results payloads.

    Groups by :func:`classify_condition`, computes per-condition stats,
    then pairwise compares each non-baseline condition against A.

    Mixed-cell ablations are refused (raises ``ValueError``) — comparing
    sparse-on-+5cm against shaped-on-+3cm is meaningless and almost
    always indicates a wrong glob.

    Args:
        payloads: Eval-results JSON dicts as loaded by
            :func:`load_eval_results`.
        bootstrap_resamples: Number of bootstrap resamples per condition.
        bootstrap_seed: Seed for the resampling RNG (deterministic
            reports).

    Returns:
        Fully-populated :class:`AblationReport`.

    Raises:
        ValueError: ``payloads`` is empty, or the runs target multiple
            perturbation cells.
    """
    if not payloads:
        raise ValueError("aggregate_runs requires at least one payload")

    signatures = {_perturbation_signature(p) for p in payloads}
    if len(signatures) > 1:
        details = ", ".join(repr(s) for s in sorted(signatures))
        raise ValueError(
            f"aggregate_runs: payloads target multiple perturbation cells "
            f"({details}); refusing to mix."
        )
    sig_kind, sig_params = next(iter(signatures))
    target_perturbation: dict[str, Any] = {"kind": sig_kind}
    for k, v in sig_params:
        target_perturbation[k] = v

    grouped: dict[str, list[tuple[float, int, str]]] = {"A": [], "B": [], "C": []}
    unknowns: list[str] = []
    for payload in payloads:
        cid, mean_tsr_c, n_roll, run_id = _condition_from_payload(payload)
        if cid == "unknown":
            unknowns.append(run_id or "<no-run-id>")
            continue
        grouped[cid].append((mean_tsr_c, n_roll, run_id))

    warnings: list[str] = []
    if unknowns:
        warnings.append(
            "unknown-condition runs ignored: " + ", ".join(sorted(unknowns))
        )

    conditions: list[ConditionStats] = []
    for cid in ("A", "B", "C"):
        runs = grouped[cid]
        if not runs:
            warnings.append(f"condition {cid} has no runs")
            continue
        means = [m for m, _, _ in runs]
        n_rollouts = sum(n for _, n, _ in runs)
        run_ids = tuple(rid for _, _, rid in runs)
        mean = fmean(means)
        std = pstdev(means) if len(means) > 1 else 0.0
        if len(means) >= 2:
            ci_lo, ci_hi = bootstrap_ci_mean(
                means,
                n_resamples=bootstrap_resamples,
                rng_seed=bootstrap_seed,
            )
        else:
            ci_lo = ci_hi = mean
            warnings.append(
                f"condition {cid} has only one run; bootstrap CI degenerate."
            )
        conditions.append(
            ConditionStats(
                condition_id=cid,
                label=CONDITION_LABELS[cid],
                n_runs=len(means),
                per_seed_means=tuple(means),
                mean=mean,
                std=std,
                bootstrap_ci_low=ci_lo,
                bootstrap_ci_high=ci_hi,
                n_rollouts=n_rollouts,
                run_ids=run_ids,
            )
        )

    condition_a = next((c for c in conditions if c.condition_id == "A"), None)
    comparisons: list[ConditionComparison] = []
    if condition_a is None:
        warnings.append("no Condition A runs; ΔTSR comparisons skipped.")
    else:
        for c in conditions:
            if c.condition_id == "A":
                continue
            t, df, p = welch_t_test(condition_a.per_seed_means, c.per_seed_means)
            comparisons.append(
                ConditionComparison(
                    condition_id=c.condition_id,
                    delta_tsr=c.mean - condition_a.mean,
                    t_statistic=t,
                    df=df,
                    p_value=p,
                    significant_at_05=(p is not None and p < SIGNIFICANCE_ALPHA),
                )
            )

    return AblationReport(
        schema_version=SCHEMA_VERSION,
        target_perturbation=target_perturbation,
        conditions=tuple(conditions),
        comparisons=tuple(comparisons),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# Serialisers                                                                 #
# --------------------------------------------------------------------------- #


def report_to_dict(report: AblationReport) -> dict[str, Any]:
    """Serialise an :class:`AblationReport` to a JSON-safe dict."""
    return {
        "schema_version": report.schema_version,
        "target_perturbation": dict(report.target_perturbation),
        "conditions": [
            {
                "condition_id": c.condition_id,
                "label": c.label,
                "n_runs": c.n_runs,
                "n_rollouts": c.n_rollouts,
                "per_seed_means": list(c.per_seed_means),
                "mean": c.mean,
                "std": c.std,
                "bootstrap_ci_low": c.bootstrap_ci_low,
                "bootstrap_ci_high": c.bootstrap_ci_high,
                "run_ids": list(c.run_ids),
            }
            for c in report.conditions
        ],
        "comparisons": [
            {
                "condition_id": comp.condition_id,
                "delta_tsr": comp.delta_tsr,
                "t_statistic": comp.t_statistic,
                "df": comp.df,
                "p_value": comp.p_value,
                "significant_at_05": comp.significant_at_05,
            }
            for comp in report.comparisons
        ],
        "warnings": list(report.warnings),
    }


def format_markdown(report: AblationReport) -> str:
    """Render a PRD §8.3 markdown table for stdout / docs.

    Two sections:

    * Per-condition: ``mean ± std`` of ``mean_tsr_custom`` with 95%
      bootstrap CI and run-count column.
    * Pairwise vs A: ΔTSR, Welch's t, df, one-sided p, significance flag.

    Trailing ``warnings`` block surfaces missing-condition / single-run
    cases so the reader can tell at a glance whether the report is
    complete.
    """
    lines: list[str] = []
    cell = report.target_perturbation
    cell_str = ", ".join(f"{k}={v}" for k, v in cell.items())
    lines.append(f"# Phase 4 Ablation -- target cell: {cell_str}")
    lines.append("")
    lines.append("## Per-condition TSR (geometric criterion)")
    lines.append("")
    lines.append(
        "| Cond | Label | n_runs | mean +/- std | 95% bootstrap CI | "
        "total rollouts |"
    )
    lines.append(
        "|------|-------|--------|--------------|------------------|"
        "----------------|"
    )
    for c in report.conditions:
        lines.append(
            f"| {c.condition_id} | {c.label} | {c.n_runs} | "
            f"{c.mean:.3f} +/- {c.std:.3f} | "
            f"[{c.bootstrap_ci_low:.3f}, {c.bootstrap_ci_high:.3f}] | "
            f"{c.n_rollouts} |"
        )
    lines.append("")
    if report.comparisons:
        lines.append("## Delta TSR vs Condition A (Welch's t, one-sided)")
        lines.append("")
        lines.append("| Cond | Delta_TSR | t | df | p | significant (a=0.05) |")
        lines.append("|------|-----------|---|----|---|----------------------|")
        for comp in report.comparisons:
            t_str = "n/a" if comp.t_statistic is None else f"{comp.t_statistic:.3f}"
            df_str = "n/a" if comp.df is None else f"{comp.df:.2f}"
            p_str = "n/a" if comp.p_value is None else f"{comp.p_value:.4f}"
            sig_str = "yes" if comp.significant_at_05 else "no"
            lines.append(
                f"| {comp.condition_id} | {comp.delta_tsr:+.3f} | "
                f"{t_str} | {df_str} | {p_str} | {sig_str} |"
            )
        lines.append("")
    if report.warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in report.warnings:
            lines.append(f"- {w}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "AblationReport",
    "BOOTSTRAP_DEFAULT_RESAMPLES",
    "BOOTSTRAP_DEFAULT_SEED",
    "CONDITION_LABELS",
    "ConditionComparison",
    "ConditionStats",
    "SCHEMA_VERSION",
    "SIGNIFICANCE_ALPHA",
    "aggregate_runs",
    "bootstrap_ci_mean",
    "classify_condition",
    "format_markdown",
    "load_eval_results",
    "report_to_dict",
    "student_t_sf",
    "welch_t_test",
]
