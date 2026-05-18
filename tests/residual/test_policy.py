"""Unit tests for ResidualMLP and ResidualCompositor."""

from __future__ import annotations

import math

import pytest
import torch

from roboeval.residual.policy import ResidualCompositor, ResidualMLP


def test_residual_mlp_forward_shape_batched():
    mlp = ResidualMLP(obs_feature_dim=64, action_dim=14)
    obs = torch.zeros(32, 64)
    base = torch.zeros(32, 14)
    out = mlp(obs, base)
    assert out.shape == (32, 14)


def test_residual_mlp_forward_shape_single():
    mlp = ResidualMLP(obs_feature_dim=8, action_dim=14)
    obs = torch.zeros(8)
    base = torch.zeros(14)
    out = mlp(obs, base)
    assert out.shape == (14,)


def test_residual_mlp_zero_obs_feature_dim_is_allowed():
    """obs_feature_dim=0 lets the residual condition only on base action."""
    mlp = ResidualMLP(obs_feature_dim=0, action_dim=14)
    obs = torch.zeros(4, 0)
    base = torch.zeros(4, 14)
    out = mlp(obs, base)
    assert out.shape == (4, 14)


def test_residual_mlp_invalid_dims_raise():
    with pytest.raises(ValueError, match="obs_feature_dim"):
        ResidualMLP(obs_feature_dim=-1, action_dim=14)
    with pytest.raises(ValueError, match="action_dim"):
        ResidualMLP(obs_feature_dim=8, action_dim=0)


def test_residual_mlp_uses_two_hidden_layers_256_gelu():
    """Spec from PRD §8.2 must be honoured exactly."""
    mlp = ResidualMLP(obs_feature_dim=64, action_dim=14, hidden_dim=256)
    layers = list(mlp.net)
    assert len(layers) == 5
    # Linear → GELU → Linear → GELU → Linear
    assert isinstance(layers[0], torch.nn.Linear)
    assert layers[0].out_features == 256
    assert isinstance(layers[1], torch.nn.GELU)
    assert isinstance(layers[2], torch.nn.Linear)
    assert layers[2].in_features == 256
    assert layers[2].out_features == 256
    assert isinstance(layers[3], torch.nn.GELU)
    assert isinstance(layers[4], torch.nn.Linear)
    assert layers[4].in_features == 256
    assert layers[4].out_features == 14


def test_compositor_initial_alpha_close_to_default():
    comp = ResidualCompositor(alpha_init=0.1)
    assert float(comp.alpha.item()) == pytest.approx(0.1, abs=1e-6)


def test_compositor_alpha_via_sigmoid_of_learnable_logit():
    """Alpha is a learned parameter; the logit must be a torch.nn.Parameter."""
    comp = ResidualCompositor(alpha_init=0.3)
    assert isinstance(comp.alpha_logit, torch.nn.Parameter)
    assert comp.alpha_logit.requires_grad is True
    # sigmoid(log(0.3/0.7)) ≈ 0.3
    assert float(comp.alpha.item()) == pytest.approx(0.3, abs=1e-6)


def test_compositor_formula_matches_prd():
    """a = clamp(a_base + alpha * a_residual, -1, 1)."""
    comp = ResidualCompositor(alpha_init=0.5)
    a_base = torch.tensor([0.3, -0.4, 0.0])
    a_res = torch.tensor([0.4, -0.2, 1.0])
    out = comp(a_base, a_res)
    expected = torch.clamp(a_base + 0.5 * a_res, -1.0, 1.0)
    assert torch.allclose(out, expected, atol=1e-6)


def test_compositor_clamps_to_action_space():
    """Residual + base outside [-1, 1] must clamp."""
    comp = ResidualCompositor(alpha_init=0.5)
    a_base = torch.tensor([0.9, -0.9])
    a_res = torch.tensor([1.0, -1.0])  # would push to 1.4, -1.4 pre-clamp
    out = comp(a_base, a_res)
    assert torch.all(out >= -1.0)
    assert torch.all(out <= 1.0)
    assert torch.allclose(out, torch.tensor([1.0, -1.0]), atol=1e-6)


def test_compositor_alpha_init_must_be_in_open_unit_interval():
    with pytest.raises(ValueError, match=r"alpha_init must be in \(0, 1\)"):
        ResidualCompositor(alpha_init=0.0)
    with pytest.raises(ValueError, match=r"alpha_init must be in \(0, 1\)"):
        ResidualCompositor(alpha_init=1.0)
    with pytest.raises(ValueError, match=r"alpha_init must be in \(0, 1\)"):
        ResidualCompositor(alpha_init=-0.5)


def test_compositor_action_bounds_validated():
    with pytest.raises(ValueError, match="action_min must be < action_max"):
        ResidualCompositor(alpha_init=0.1, action_min=1.0, action_max=-1.0)


def test_compositor_alpha_logit_is_consistent_with_alpha_init():
    """sigmoid(logit) round-trips to the requested alpha_init."""
    for alpha_init in [0.05, 0.1, 0.25, 0.5, 0.75, 0.95]:
        comp = ResidualCompositor(alpha_init=alpha_init)
        assert float(comp.alpha.item()) == pytest.approx(alpha_init, abs=1e-6)
        # Logit math is correct:
        expected_logit = math.log(alpha_init / (1.0 - alpha_init))
        assert float(comp.alpha_logit.item()) == pytest.approx(expected_logit, abs=1e-6)


def test_compositor_alpha_responds_to_logit_update():
    """An optimiser step on alpha_logit should change alpha (regression check)."""
    comp = ResidualCompositor(alpha_init=0.1)
    alpha_before = float(comp.alpha.item())
    comp.alpha_logit.data += 1.0
    alpha_after = float(comp.alpha.item())
    assert alpha_after > alpha_before
