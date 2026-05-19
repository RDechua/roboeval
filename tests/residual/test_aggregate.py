"""Unit tests for the Phase 4 ablation aggregator.

Three layers:

1. **Stats math** — the stdlib Student-t SF and Welch's t-test
   implementations are validated against hand-computed reference
   values (tolerance 1e-3, matches `scipy.stats.t.sf` to that
   precision at df ∈ {1, 4, 8, 1000} and at the tail values that
   actually drive 0.05 significance decisions).
2. **Bootstrap CI** — determinism with a fixed seed; bounds make sense
   on synthetic inputs.
3. **End-to-end aggregation** — synthetic payloads exercise the full
   classify→group→stats→serialise→markdown path, including missing
   conditions, single-run conditions, unknown payloads, and mixed-cell
   refusal.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from roboeval.residual.aggregate import (
    SIGNIFICANCE_ALPHA,
    AblationReport,
    aggregate_runs,
    bootstrap_ci_mean,
    classify_condition,
    format_markdown,
    load_eval_results,
    report_to_dict,
    student_t_sf,
    welch_t_test,
)

# --------------------------------------------------------------------------- #
# Student's t survival function                                               #
# --------------------------------------------------------------------------- #


class TestStudentTSF:
    def test_t_zero_is_half(self) -> None:
        # By symmetry, SF(0) = 0.5 for any df.
        for df in (1, 2, 4, 8, 1000):
            assert student_t_sf(0.0, df) == pytest.approx(0.5)

    def test_negative_symmetric(self) -> None:
        # sf(-t, df) = 1 - sf(t, df).
        assert student_t_sf(-1.5, 4) == pytest.approx(1.0 - student_t_sf(1.5, 4))

    def test_cauchy_df1(self) -> None:
        # df=1 is the Cauchy distribution: SF(1) = 0.25 exactly.
        assert student_t_sf(1.0, 1) == pytest.approx(0.25, abs=1e-4)

    def test_critical_values_match_table(self) -> None:
        # Standard one-sided critical values from any stats table:
        # df=4, t=2.776 → p ≈ 0.025
        # df=8, t=2.306 → p ≈ 0.025
        # df=1000, t=1.960 → p ≈ 0.025 (≈ normal)
        # df=4, t=2.132 → p ≈ 0.050
        assert student_t_sf(2.776, 4) == pytest.approx(0.025, abs=1e-3)
        assert student_t_sf(2.306, 8) == pytest.approx(0.025, abs=1e-3)
        assert student_t_sf(1.960, 1000) == pytest.approx(0.025, abs=1e-3)
        assert student_t_sf(2.132, 4) == pytest.approx(0.050, abs=1e-3)

    def test_large_t_goes_to_zero(self) -> None:
        # Far tail.
        assert student_t_sf(20.0, 4) < 1e-4
        assert student_t_sf(50.0, 4) < 1e-6

    def test_rejects_nonpositive_df(self) -> None:
        with pytest.raises(ValueError, match="df must be positive"):
            student_t_sf(1.0, 0.0)


# --------------------------------------------------------------------------- #
# Welch's t-test                                                              #
# --------------------------------------------------------------------------- #


class TestWelchTTest:
    def test_identical_arms_zero_t(self) -> None:
        # Same values both sides → t=0, p=0.5 (one-sided).
        t, df, p = welch_t_test([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        assert t == pytest.approx(0.0)
        assert p == pytest.approx(0.5)
        assert df == pytest.approx(4.0)

    def test_clear_separation_significant(self) -> None:
        # Large effect: 0.2 vs 0.8 with low variance → p << 0.05.
        t, df, p = welch_t_test(
            [0.18, 0.20, 0.22],
            [0.78, 0.80, 0.82],
        )
        assert t is not None and p is not None and df is not None
        assert t > 5.0
        assert p < 0.005

    def test_handcomputed_reference(self) -> None:
        # Hand-computed reference (verified offline against scipy):
        # A = [0.6, 0.7, 0.5] → mean=0.6, sample var=0.01
        # B = [0.7, 0.8, 0.9] → mean=0.8, sample var=0.01
        # se = sqrt(0.01/3 + 0.01/3) = sqrt(0.006666...) ≈ 0.08165
        # t = (0.8 - 0.6) / 0.08165 ≈ 2.4495
        # df = (2/3 * 0.01)^2 / (2 * (0.01/3)^2 / 2) = 4
        # scipy.stats.t.sf(2.4495, 4) ≈ 0.0353
        t, df, p = welch_t_test([0.6, 0.7, 0.5], [0.7, 0.8, 0.9])
        assert t is not None and df is not None and p is not None
        assert t == pytest.approx(2.4495, abs=1e-3)
        assert df == pytest.approx(4.0, abs=1e-3)
        assert p == pytest.approx(0.0353, abs=2e-3)

    def test_too_few_samples_returns_none(self) -> None:
        assert welch_t_test([0.5], [0.5, 0.5]) == (None, None, None)
        assert welch_t_test([], [0.5, 0.5]) == (None, None, None)

    def test_zero_variance_same_mean_returns_half(self) -> None:
        t, df, p = welch_t_test([0.5, 0.5], [0.5, 0.5])
        assert t == 0.0
        assert p == pytest.approx(0.5)

    def test_zero_variance_different_mean_returns_none(self) -> None:
        # SE undefined → t undefined.
        assert welch_t_test([0.4, 0.4], [0.6, 0.6]) == (None, None, None)

    def test_b_less_than_a_gives_p_above_half(self) -> None:
        # One-sided H1: B > A. If B < A then p > 0.5.
        _, _, p = welch_t_test([0.8, 0.7, 0.9], [0.2, 0.3, 0.1])
        assert p is not None and p > 0.5


# --------------------------------------------------------------------------- #
# Bootstrap CI                                                                #
# --------------------------------------------------------------------------- #


class TestBootstrapCI:
    def test_deterministic_with_seed(self) -> None:
        vals = [0.5, 0.6, 0.7]
        ci1 = bootstrap_ci_mean(vals, n_resamples=500, rng_seed=42)
        ci2 = bootstrap_ci_mean(vals, n_resamples=500, rng_seed=42)
        assert ci1 == ci2

    def test_ci_contains_mean(self) -> None:
        vals = [0.4, 0.5, 0.6, 0.5, 0.55]
        lo, hi = bootstrap_ci_mean(vals, n_resamples=2000, rng_seed=0)
        mean = sum(vals) / len(vals)
        assert lo <= mean <= hi

    def test_ci_tightens_as_data_concentrates(self) -> None:
        wide = bootstrap_ci_mean([0.1, 0.5, 0.9], n_resamples=2000, rng_seed=0)
        tight = bootstrap_ci_mean([0.49, 0.50, 0.51], n_resamples=2000, rng_seed=0)
        assert (wide[1] - wide[0]) > (tight[1] - tight[0])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one value"):
            bootstrap_ci_mean([])


# --------------------------------------------------------------------------- #
# Classification                                                              #
# --------------------------------------------------------------------------- #


class TestClassifyCondition:
    def test_condition_a(self) -> None:
        assert classify_condition({"policy_kind": "act"}) == "A"
        assert classify_condition({"policy_kind": "act", "residual": None}) == "A"

    def test_condition_b(self) -> None:
        payload = {
            "policy_kind": "residual_act",
            "residual": {"reward_kind": "sparse", "alpha_init": 0.05},
        }
        assert classify_condition(payload) == "B"

    def test_condition_c(self) -> None:
        payload = {
            "policy_kind": "residual_act",
            "residual": {"reward_kind": "shaped", "alpha_init": 0.05},
        }
        assert classify_condition(payload) == "C"

    def test_unknown_combinations(self) -> None:
        # Residual policy without reward_kind block.
        assert classify_condition({"policy_kind": "residual_act"}) == "unknown"
        # Residual with unknown reward kind.
        assert (
            classify_condition(
                {
                    "policy_kind": "residual_act",
                    "residual": {"reward_kind": "exotic"},
                }
            )
            == "unknown"
        )
        # ACT with a residual block (schema violation).
        assert (
            classify_condition(
                {"policy_kind": "act", "residual": {"reward_kind": "sparse"}}
            )
            == "unknown"
        )
        # Totally empty.
        assert classify_condition({}) == "unknown"


# --------------------------------------------------------------------------- #
# End-to-end aggregation                                                      #
# --------------------------------------------------------------------------- #


def _make_payload(
    *,
    condition: str,
    run_id: str,
    mean_tsr_custom: float,
    n_rollouts: int = 150,
    dy_m: float = 0.05,
) -> dict[str, Any]:
    """Build a minimal eval-results payload for one run.

    Only the fields the aggregator reads are populated; the full
    per-rollout dump is intentionally omitted to keep the test fast.
    """
    base: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "perturbation_kind": "spatial",
        "perturbation_params": {"dx_m": 0.0, "dy_m": dy_m},
        "metrics": {
            "mean_tsr_custom": mean_tsr_custom,
            "n_rollouts": n_rollouts,
            "n_seed_groups": 3,
            "per_seed_tsr_custom": [
                mean_tsr_custom - 0.05,
                mean_tsr_custom,
                mean_tsr_custom + 0.05,
            ],
        },
    }
    if condition == "A":
        base["policy_kind"] = "act"
    else:
        base["policy_kind"] = "residual_act"
        base["residual"] = {
            "reward_kind": "sparse" if condition == "B" else "shaped",
            "alpha_init": 0.05,
            "log_std_init": -2.0,
        }
    return base


def _three_seed_set(condition: str, means: list[float]) -> list[dict[str, Any]]:
    return [
        _make_payload(
            condition=condition,
            run_id=f"{condition.lower()}_seed{i}",
            mean_tsr_custom=m,
        )
        for i, m in enumerate(means)
    ]


class TestAggregateRuns:
    def test_full_3x3_ablation(self) -> None:
        payloads = (
            _three_seed_set("A", [0.30, 0.32, 0.34])
            + _three_seed_set("B", [0.55, 0.60, 0.50])
            + _three_seed_set("C", [0.65, 0.70, 0.60])
        )
        report = aggregate_runs(payloads, bootstrap_resamples=500)

        assert report.target_perturbation == {
            "kind": "spatial",
            "dx_m": 0.0,
            "dy_m": 0.05,
        }
        assert len(report.conditions) == 3
        ids = [c.condition_id for c in report.conditions]
        assert ids == ["A", "B", "C"]

        cond_a, cond_b, cond_c = report.conditions
        assert cond_a.mean == pytest.approx(0.32)
        assert cond_b.mean == pytest.approx(0.55)
        assert cond_c.mean == pytest.approx(0.65)

        # Two comparisons (B vs A, C vs A).
        assert [c.condition_id for c in report.comparisons] == ["B", "C"]
        b_vs_a, c_vs_a = report.comparisons
        assert b_vs_a.delta_tsr == pytest.approx(0.23, abs=1e-6)
        assert c_vs_a.delta_tsr == pytest.approx(0.33, abs=1e-6)
        # B vs A: clearly large effect.
        assert b_vs_a.p_value is not None and b_vs_a.p_value < SIGNIFICANCE_ALPHA
        assert b_vs_a.significant_at_05 is True
        # C vs A: even larger.
        assert c_vs_a.p_value is not None and c_vs_a.p_value < SIGNIFICANCE_ALPHA
        assert c_vs_a.significant_at_05 is True
        assert report.warnings == ()

    def test_null_result_not_significant(self) -> None:
        # B improves only marginally, noise dominates → not significant.
        payloads = _three_seed_set("A", [0.30, 0.35, 0.40]) + _three_seed_set(
            "B", [0.32, 0.36, 0.41]
        )
        report = aggregate_runs(payloads, bootstrap_resamples=500)
        b_vs_a = report.comparisons[0]
        assert b_vs_a.condition_id == "B"
        assert b_vs_a.significant_at_05 is False
        assert b_vs_a.p_value is not None and b_vs_a.p_value > 0.05

    def test_missing_condition_a_skips_comparisons(self) -> None:
        payloads = _three_seed_set("B", [0.5, 0.55, 0.6])
        report = aggregate_runs(payloads, bootstrap_resamples=500)
        assert report.comparisons == ()
        assert any("no Condition A" in w for w in report.warnings)

    def test_single_run_condition_warns(self) -> None:
        payloads = _three_seed_set("A", [0.3, 0.32, 0.34]) + _three_seed_set("B", [0.6])
        report = aggregate_runs(payloads, bootstrap_resamples=500)
        # B is present but single-run → stat math can't compute t.
        b_vs_a = report.comparisons[0]
        assert b_vs_a.t_statistic is None
        assert b_vs_a.significant_at_05 is False
        assert any("only one run" in w for w in report.warnings)

    def test_unknown_payloads_surface_as_warning(self) -> None:
        good = _three_seed_set("A", [0.3, 0.35, 0.4])
        bad = [
            {
                "schema_version": 1,
                "run_id": "broken_run",
                "perturbation_kind": "spatial",
                "perturbation_params": {"dx_m": 0.0, "dy_m": 0.05},
                "policy_kind": "diffusion_policy",  # unsupported
                "metrics": {"mean_tsr_custom": 0.0, "n_rollouts": 0},
            }
        ]
        report = aggregate_runs(good + bad, bootstrap_resamples=200)
        assert any("unknown-condition" in w for w in report.warnings)
        # The known runs are still aggregated.
        assert report.conditions[0].n_runs == 3

    def test_mixed_cells_refused(self) -> None:
        a_at_5cm = _three_seed_set("A", [0.3, 0.32, 0.34])
        b_at_3cm = [
            _make_payload(
                condition="B", run_id=f"b_seed{i}", mean_tsr_custom=0.5, dy_m=0.03
            )
            for i in range(3)
        ]
        with pytest.raises(ValueError, match="multiple perturbation cells"):
            aggregate_runs(a_at_5cm + b_at_3cm)

    def test_empty_payloads_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one payload"):
            aggregate_runs([])


# --------------------------------------------------------------------------- #
# Loader + serialisers                                                        #
# --------------------------------------------------------------------------- #


class TestLoaderAndSerialisers:
    def test_load_round_trips(self, tmp_path: Path) -> None:
        payload = _make_payload(condition="A", run_id="r1", mean_tsr_custom=0.3)
        path = tmp_path / "eval_results_r1.json"
        path.write_text(json.dumps(payload))
        loaded = load_eval_results([path])
        assert len(loaded) == 1
        assert loaded[0]["run_id"] == "r1"
        assert loaded[0]["metrics"]["mean_tsr_custom"] == pytest.approx(0.3)

    def test_load_preserves_order(self, tmp_path: Path) -> None:
        paths: list[Path] = []
        for i in range(3):
            payload = _make_payload(
                condition="A", run_id=f"r{i}", mean_tsr_custom=0.3 + i * 0.05
            )
            p = tmp_path / f"eval_results_r{i}.json"
            p.write_text(json.dumps(payload))
            paths.append(p)
        loaded = load_eval_results(paths)
        assert [p["run_id"] for p in loaded] == ["r0", "r1", "r2"]

    def test_report_to_dict_round_trips_through_json(self) -> None:
        payloads = (
            _three_seed_set("A", [0.3, 0.32, 0.34])
            + _three_seed_set("B", [0.5, 0.55, 0.6])
            + _three_seed_set("C", [0.6, 0.65, 0.7])
        )
        report = aggregate_runs(payloads, bootstrap_resamples=200)
        d = report_to_dict(report)
        # JSON-safe.
        serialised = json.dumps(d)
        rehydrated = json.loads(serialised)
        assert rehydrated["schema_version"] == 1
        assert len(rehydrated["conditions"]) == 3
        assert len(rehydrated["comparisons"]) == 2
        assert rehydrated["target_perturbation"]["dy_m"] == pytest.approx(0.05)

    def test_format_markdown_contains_key_sections(self) -> None:
        payloads = _three_seed_set("A", [0.3, 0.32, 0.34]) + _three_seed_set(
            "B", [0.5, 0.55, 0.6]
        )
        report = aggregate_runs(payloads, bootstrap_resamples=200)
        md = format_markdown(report)
        assert "Phase 4 Ablation" in md
        assert "Per-condition TSR" in md
        assert "Delta TSR vs Condition A" in md
        # Both contributing condition labels show up.
        assert "Frozen base only" in md
        assert "Residual RL, sparse reward" in md

    def test_format_markdown_warnings_section_omitted_when_clean(self) -> None:
        payloads = (
            _three_seed_set("A", [0.3, 0.32, 0.34])
            + _three_seed_set("B", [0.5, 0.55, 0.6])
            + _three_seed_set("C", [0.6, 0.65, 0.7])
        )
        report = aggregate_runs(payloads, bootstrap_resamples=200)
        md = format_markdown(report)
        assert "## Warnings" not in md

    def test_format_markdown_shows_warnings_when_present(self) -> None:
        # Missing Condition A → warning section.
        payloads = _three_seed_set("B", [0.5, 0.55, 0.6])
        report = aggregate_runs(payloads, bootstrap_resamples=200)
        md = format_markdown(report)
        assert "## Warnings" in md
        assert "no Condition A" in md


# --------------------------------------------------------------------------- #
# AblationReport dataclass surface                                            #
# --------------------------------------------------------------------------- #


def test_report_is_frozen_dataclass() -> None:
    """Catch accidental refactors that drop the frozen=True / slots=True."""
    payloads = _three_seed_set("A", [0.3, 0.32, 0.34])
    report = aggregate_runs(payloads, bootstrap_resamples=200)
    assert isinstance(report, AblationReport)
    with pytest.raises((AttributeError, TypeError)):
        report.schema_version = 999  # type: ignore[misc]
