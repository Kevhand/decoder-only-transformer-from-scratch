"""Tests for Positional_Encoding/rotary_positional_encoding.py.
Run from the project root:  python -m tests.test_rope"""

import math

import torch

from Positional_Encoding.rotary_positional_encoding import ROPE

torch.manual_seed(42)
D, MAX = 8, 32


def expect_error(fn, message):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError(message)


def test_hand_computation():
    # One vector at position 5: each (even, odd) pair is rotated by 5 * 10000^(-i/D).
    rope = ROPE(D, MAX)
    q = torch.randn(1, 1, 1, D)
    q_rot, _ = rope(q, q, q_offset=5, k_offset=5)
    for pair in range(D // 2):
        angle = 5 * 10000 ** (-(2 * pair) / D)
        a, b = q[0, 0, 0, 2 * pair].item(), q[0, 0, 0, 2 * pair + 1].item()
        expected = (a * math.cos(angle) - b * math.sin(angle), a * math.sin(angle) + b * math.cos(angle))
        got = (q_rot[0, 0, 0, 2 * pair].item(), q_rot[0, 0, 0, 2 * pair + 1].item())
        assert all(abs(e - g) < 1e-5 for e, g in zip(expected, got)), f"pair {pair} wrong"
    print("PASS  each (even, odd) pair rotated by position x frequency (hand computation)")


def test_rotation_preserves_length():
    rope = ROPE(D, MAX)
    q = torch.randn(2, 3, 10, D)
    q_rot, _ = rope(q, q)
    assert torch.allclose(q.norm(dim=-1), q_rot.norm(dim=-1), atol=1e-5)
    print("PASS  rotation preserves vector length")


def test_position_zero_is_identity():
    rope = ROPE(D, MAX)
    q, k = torch.randn(1, 2, 1, D), torch.randn(1, 2, 1, D)
    q_rot, k_rot = rope(q, k)
    assert torch.allclose(q_rot, q, atol=1e-6) and torch.allclose(k_rot, k, atol=1e-6)
    print("PASS  position 0 leaves q and k unchanged")


def test_relative_position():
    rope = ROPE(D, MAX)
    q, k = torch.randn(1, 1, 1, D), torch.randn(1, 1, 1, D)

    def score(qp, kp):
        qr, kr = rope(q, k, q_offset=qp, k_offset=kp)
        return (qr * kr).sum().item()

    same_gap = abs(score(3, 1) - score(10, 8))
    different_gap = abs(score(3, 1) - score(5, 1))
    assert same_gap < 1e-5, "score should depend only on the position difference"
    assert different_gap > 1e-3, "negative control: a different gap should change the score"
    print(f"PASS  score depends only on relative position (same gap diff {same_gap:.1e}, "
          f"different gap diff {different_gap:.3f})")


def test_cached_equals_full():
    rope = ROPE(D, MAX)
    q, k = torch.randn(2, 2, 12, D), torch.randn(2, 2, 12, D)
    q_full, k_full = rope(q, k)

    # Token by token
    pieces = [rope(q[:, :, i:i + 1], k[:, :, i:i + 1], q_offset=i, k_offset=i) for i in range(12)]
    assert torch.allclose(q_full, torch.cat([p[0] for p in pieces], 2), atol=1e-6)
    assert torch.allclose(k_full, torch.cat([p[1] for p in pieces], 2), atol=1e-6)

    # Uneven chunks
    pieces = [rope(q[:, :, s:e], k[:, :, s:e], q_offset=s, k_offset=s) for s, e in [(0, 3), (3, 7), (7, 12)]]
    assert torch.allclose(q_full, torch.cat([p[0] for p in pieces], 2), atol=1e-6)
    assert torch.allclose(k_full, torch.cat([p[1] for p in pieces], 2), atol=1e-6)
    print("PASS  token-by-token and chunked offsets match a full pass")


def test_errors():
    rope = ROPE(D, MAX)
    two = torch.randn(1, 1, 2, D)
    expect_error(lambda: rope(two, two, q_offset=MAX - 1, k_offset=0), "Q overflow not caught")
    expect_error(lambda: rope(two, two, q_offset=0, k_offset=MAX - 1), "K overflow not caught")
    expect_error(lambda: rope(two, two, q_offset=-1, k_offset=0), "negative offset not caught")
    rope(two, two, q_offset=MAX - 2, k_offset=MAX - 2)  # exactly at the limit is fine
    try:
        ROPE(7, MAX)
    except AssertionError:
        pass
    else:
        raise AssertionError("odd head_dim not rejected")
    print("PASS  Q overflow, K overflow, negative offset and odd head_dim are rejected; the limit itself works")


def test_buffers_not_saved():
    rope = ROPE(D, MAX)
    assert len(rope.state_dict()) == 0, "cos/sin tables should not be in checkpoints"
    assert len(list(rope.buffers())) == 2
    print("PASS  cos/sin tables are buffers but are not saved in state_dict")


def test_bf16_inputs_rotate_in_float32():
    rope = ROPE(D, MAX)
    q = torch.randn(1, 1, 4, D)
    q_rot, _ = rope(q.bfloat16(), q.bfloat16())
    reference, _ = rope(q.bfloat16().float(), q.bfloat16().float())
    assert q_rot.dtype == torch.float32
    assert torch.allclose(q_rot, reference, atol=1e-6)
    print("PASS  bf16 inputs are rotated in float32 (tables are float32), as intended for mixed precision")


if __name__ == "__main__":
    test_hand_computation()
    test_rotation_preserves_length()
    test_position_zero_is_identity()
    test_relative_position()
    test_cached_equals_full()
    test_errors()
    test_buffers_not_saved()
    test_bf16_inputs_rotate_in_float32()
    print("\nAll RoPE tests passed.")