"""Tests for FeedForward_Network/ffn.py. Run from the project root:  python -m tests.test_ffn"""

import torch
import torch.nn.functional as F

from FeedForward_Network.ffn import FFN

torch.manual_seed(42)


def test_shape_and_parameters():
    ffn = FFN(16, 64)
    assert ffn(torch.randn(2, 5, 16)).shape == (2, 5, 16)
    assert sum(p.numel() for p in ffn.parameters()) == 16 * 64 + 64 + 64 * 16 + 16
    print("PASS  output shape matches input; parameter count = 2 matrices + 2 biases")


def test_hand_computation():
    ffn = FFN(16, 64)
    x = torch.randn(2, 5, 16)
    with torch.no_grad():
        expected = F.gelu(x @ ffn.linear1.weight.T + ffn.linear1.bias) @ ffn.linear2.weight.T + ffn.linear2.bias
        assert torch.allclose(ffn(x), expected, atol=1e-6)
    print("PASS  matches linear2(gelu(linear1(x))) computed by hand")


def test_positions_independent():
    ffn = FFN(16, 64)
    x = torch.randn(2, 5, 16)
    changed = x.clone()
    changed[0, 2] += 10.0
    with torch.no_grad():
        a, b = ffn(x), ffn(changed)
    assert not torch.allclose(a[0, 2], b[0, 2]), "changed position did not change"
    assert torch.equal(a[0, [0, 1, 3, 4]], b[0, [0, 1, 3, 4]]), "other positions changed"
    assert torch.equal(a[1], b[1]), "other sequence in the batch changed"
    print("PASS  each position processed independently (with negative control)")


def test_gradients_flow():
    ffn = FFN(16, 64)
    x = torch.randn(2, 5, 16, requires_grad=True)
    (ffn(x) * torch.randn(2, 5, 16)).sum().backward()
    for name, p in ffn.named_parameters():
        assert p.grad is not None and torch.count_nonzero(p.grad) > 0, f"{name} got no gradient"
    assert torch.count_nonzero(x.grad) > 0
    print("PASS  all parameters and the input receive gradients")


if __name__ == "__main__":
    test_shape_and_parameters()
    test_hand_computation()
    test_positions_independent()
    test_gradients_flow()
    print("\nAll FFN tests passed.")