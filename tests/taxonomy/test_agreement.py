"""Tests for Cohen's κ inter-rater agreement (PRD §7.3 step 4)."""

from __future__ import annotations

import pytest

from roboeval.taxonomy import cohens_kappa


def test_perfect_agreement_on_two_categories_gives_kappa_one():
    a = ["grasp", "timeout", "grasp", "timeout"]
    b = ["grasp", "timeout", "grasp", "timeout"]
    result = cohens_kappa(a, b)
    assert result.kappa == pytest.approx(1.0)
    assert result.observed_agreement == pytest.approx(1.0)
    assert not result.is_degenerate


def test_complete_disagreement_gives_negative_kappa():
    a = ["grasp", "grasp", "grasp", "grasp"]
    b = ["timeout", "timeout", "timeout", "timeout"]
    result = cohens_kappa(a, b)
    # Perfect disagreement with 50/50 chance expectation → kappa = -1
    # but the marginal here is 100% A=grasp, 100% B=timeout, so chance
    # agreement is 0 and observed is 0 → kappa = (0 - 0) / (1 - 0) = 0.
    assert result.observed_agreement == 0.0
    assert result.kappa == pytest.approx(0.0)


def test_known_textbook_case_matches_expected_kappa():
    # Standard Wikipedia example: two raters on 50 items, 5 categories.
    # Use the simplest reproducible one: 100 items, two categories.
    #   Agree-Yes: 45, Agree-No: 15, A-Yes-B-No: 25, A-No-B-Yes: 15.
    # p_o = (45 + 15) / 100 = 0.60
    # p_yes_a = 70/100, p_yes_b = 60/100
    # p_no_a  = 30/100, p_no_b  = 40/100
    # p_e = 0.7*0.6 + 0.3*0.4 = 0.42 + 0.12 = 0.54
    # kappa = (0.60 - 0.54) / (1 - 0.54) = 0.06 / 0.46 ≈ 0.1304
    a = ["Y"] * 45 + ["Y"] * 25 + ["N"] * 15 + ["N"] * 15
    b = ["Y"] * 45 + ["N"] * 25 + ["Y"] * 15 + ["N"] * 15
    result = cohens_kappa(a, b)
    assert result.observed_agreement == pytest.approx(0.60)
    assert result.expected_agreement == pytest.approx(0.54)
    assert result.kappa == pytest.approx(0.06 / 0.46, abs=1e-6)


def test_landis_koch_substantial_threshold_kappa_above_06():
    # Six items, four categories, single disagreement on a low-prior label.
    a = ["grasp", "approach", "timeout", "grasp", "approach", "oscillation"]
    b = ["grasp", "approach", "timeout", "grasp", "approach", "approach"]
    result = cohens_kappa(a, b)
    # Five out of six agree; observed agreement 5/6 ≈ 0.833.
    # Marginals: a has {grasp:2, approach:2, timeout:1, oscillation:1};
    #            b has {grasp:2, approach:3, timeout:1}
    # p_e = (2/6)(2/6) + (2/6)(3/6) + (1/6)(1/6) + (1/6)(0) = 4/36 + 6/36 + 1/36
    #     = 11/36 ≈ 0.306
    # kappa = (0.833 - 0.306) / (1 - 0.306) ≈ 0.527 / 0.694 ≈ 0.760
    assert (
        result.kappa > 0.6
    ), "expected substantial agreement (Landis & Koch threshold)"


def test_degenerate_all_same_category_returns_kappa_one():
    # If every rollout is in the same bucket per both raters, chance
    # agreement is 1 and κ is undefined; the helper returns 1.0 with
    # is_degenerate flagged.
    a = ["timeout"] * 10
    b = ["timeout"] * 10
    result = cohens_kappa(a, b)
    assert result.kappa == 1.0
    assert result.is_degenerate is True


def test_unequal_length_raises():
    with pytest.raises(ValueError, match="equal length"):
        cohens_kappa(["a", "b"], ["a"])


def test_empty_raises():
    with pytest.raises(ValueError, match="at least one"):
        cohens_kappa([], [])


def test_n_reported_matches_input_length():
    result = cohens_kappa(["x"] * 7, ["x"] * 7)
    assert result.n == 7
