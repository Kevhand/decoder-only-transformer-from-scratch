"""Tests for LayerNorm/layer_norm.py. Run from the project root:  python -m tests.test_layer_norm"""

import torch
import torch.nn as nn

from LayerNorm.layer_norm import LayerNorm

torch.manual_seed(42)


def make_pair(d_model, random_params):
    ours = LayerNorm(d_model, eps=1e-5)
    ref = nn.LayerNorm(d_model, eps=1e-5)
    if random_params:
        with torch.no_grad():
            ours.gamma.copy_(torch.randn(d_model))
            ours.beta.copy_(torch.randn(d_model))
    with torch.no_grad():
        ref.weight.copy_(ours.gamma)
        ref.bias.copy_(ours.beta)
    return ours, ref


def test_initial_parameters():
    ln = LayerNorm(8)
    assert torch.equal(ln.gamma, torch.ones(8)) and torch.equal(ln.beta, torch.zeros(8))
    assert sum(p.numel() for p in ln.parameters()) == 16
    print("PASS  gamma starts at ones, beta at zeros, 2 x d_model parameters")


def test_normalizes_each_position():
    ln = LayerNorm(16)
    x = torch.randn(3, 7, 16) * 5 + 3
    with torch.no_grad():
        y = ln(x)
    var_in = x.var(dim=-1, unbiased=False)
    # With gamma=1, beta=0: mean 0, variance var/(var+eps) at every position (not exactly 1).
    assert torch.allclose(y.mean(dim=-1), torch.zeros(3, 7), atol=1e-5)
    assert torch.allclose(y.var(dim=-1, unbiased=False), var_in / (var_in + 1e-5), atol=1e-5)
    print("PASS  every position normalized over the last dimension: mean 0, variance var/(var+eps)")


def test_matches_pytorch(random_params):
    ours, ref = make_pair(16, random_params)
    x = torch.randn(4, 6, 16)
    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)

    y1, y2 = ours(x1), ref(x2)
    assert torch.allclose(y1, y2, atol=1e-6), "forward differs from nn.LayerNorm"

    # A weighted-sum loss: unlike sum(y^2), it doesn't cancel out under normalization.
    probe = torch.randn_like(x)
    (y1 * probe).sum().backward()
    (y2 * probe).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5), "input gradient differs"
    assert torch.allclose(ours.gamma.grad, ref.weight.grad, atol=1e-5), "gamma gradient differs"
    assert torch.allclose(ours.beta.grad, ref.bias.grad, atol=1e-5), "beta gradient differs"
    label = "random gamma/beta" if random_params else "initial gamma/beta"
    print(f"PASS  matches nn.LayerNorm in forward and all three gradients ({label})")


if __name__ == "__main__":
    test_initial_parameters()
    test_normalizes_each_position()
    test_matches_pytorch(random_params=False)
    test_matches_pytorch(random_params=True)
    print("\nAll LayerNorm tests passed.")