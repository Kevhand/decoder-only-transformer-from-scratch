"""Tests for Attention_Mechanism/multi_head_causal_attention.py.
Run from the project root:  python -m tests.test_attention"""

from contextlib import contextmanager

import torch

from Attention_Mechanism.multi_head_causal_attention import MultiHeadCausalAttention

torch.manual_seed(42)
B, H, T, HD = 2, 4, 10, 8
D = H * HD


@contextmanager
def sdpa(enabled):
    old = MultiHeadCausalAttention.use_sdpa
    MultiHeadCausalAttention.use_sdpa = enabled
    try:
        yield
    finally:
        MultiHeadCausalAttention.use_sdpa = old


def qkv(seq=T):
    return torch.randn(B, H, seq, HD), torch.randn(B, H, seq, HD), torch.randn(B, H, seq, HD)


def test_manual_weights():
    attn = MultiHeadCausalAttention(D, 64)
    with sdpa(False), torch.no_grad():
        out, w = attn(*qkv())
    assert out.shape == (B, T, D) and w.shape == (B, H, T, T)
    assert torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=1e-6), "rows must sum to 1"
    assert torch.count_nonzero(torch.triu(w, diagonal=1)) == 0, "future positions must get zero weight"
    print("PASS  manual path: weights shaped (B, H, T, T), rows sum to 1, future weights exactly 0")


def test_sdpa_matches_manual():
    attn = MultiHeadCausalAttention(D, 64)
    Q, K, V = [t.requires_grad_(True) for t in qkv()]
    probe = torch.randn(B, T, D)
    results = {}
    for enabled in (False, True):
        for t in (Q, K, V):
            t.grad = None
        attn.zero_grad()
        with sdpa(enabled):
            out, w = attn(Q, K, V)
        (out * probe).sum().backward()
        results[enabled] = (out.detach(), [t.grad.clone() for t in (Q, K, V)], attn.W_o.weight.grad.clone(), w)

    (o_m, g_m, w_m, _), (o_s, g_s, w_s, weights) = results[False], results[True]
    assert weights is None, "fused path returns no attention weights"
    assert torch.allclose(o_m, o_s, atol=1e-5), "forward differs"
    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(g_m, g_s)), "Q/K/V gradients differ"
    assert torch.allclose(w_m, w_s, atol=1e-5), "W_o gradient differs"
    print(f"PASS  fused path matches manual: forward diff {(o_m - o_s).abs().max():.1e}, "
          f"gradients match; fused returns weights=None")


def test_causality():
    attn = MultiHeadCausalAttention(D, 64)
    Q, K, V = qkv()
    K2, V2 = K.clone(), V.clone()
    K2[:, :, 6] += 5.0
    V2[:, :, 6] += 5.0
    for enabled in (False, True):
        with sdpa(enabled), torch.no_grad():
            a, _ = attn(Q, K, V)
            b, _ = attn(Q, K2, V2)
        assert torch.equal(a[:, :6], b[:, :6]), "a future key/value changed an earlier output"
        for pos in range(6, T):
            assert not torch.allclose(a[:, pos], b[:, pos]), f"position {pos} should see the change"
    print("PASS  causality in both paths: earlier outputs unchanged, positions 6..9 each changed")


def test_cache_equivalence():
    attn = MultiHeadCausalAttention(D, 64)
    Q, K, V = qkv()
    for enabled in (False, True):
        with sdpa(enabled), torch.no_grad():
            full, _ = attn(Q, K, V)
            for chunks in ([(i, i + 1) for i in range(T)], [(0, 3), (3, 7), (7, 10)]):
                outs = [attn(Q[:, :, s:e], K[:, :, :e], V[:, :, :e], past_len=s)[0] for s, e in chunks]
                assert torch.allclose(full, torch.cat(outs, 1), atol=1e-5)
    print("PASS  token-by-token and chunked (rectangular mask) match a full pass in both paths")


def test_cache_mismatch_rejected():
    attn = MultiHeadCausalAttention(D, 64)
    Q, K, V = qkv()
    try:
        attn(Q[:, :, 5:6], K[:, :, :5], V[:, :, :5], past_len=5)  # 5 keys, but 6 expected
    except ValueError:
        print("PASS  a KV cache of the wrong length is rejected")
        return
    raise AssertionError("expected a ValueError for a mismatched cache")


def test_mixed_precision():
    # Under bf16 autocast, RoPE returns float32 Q and K while V is bf16.
    attn = MultiHeadCausalAttention(D, 64)
    Q, K, V = qkv()
    with torch.no_grad():
        with sdpa(False):
            reference, _ = attn(Q, K, V)
        for enabled in (False, True):
            with sdpa(enabled), torch.autocast("cpu", dtype=torch.bfloat16):
                out, _ = attn(Q, K, V.bfloat16())
            assert torch.isfinite(out).all()
            assert (out.float() - reference).abs().max() < 0.05, "bf16 result far from float32 reference"
    print("PASS  float32 Q/K with bf16 V works under bf16 autocast in both paths")


if __name__ == "__main__":
    test_manual_weights()
    test_sdpa_matches_manual()
    test_causality()
    test_cache_equivalence()
    test_cache_mismatch_rejected()
    test_mixed_precision()
    print("\nAll attention tests passed.")