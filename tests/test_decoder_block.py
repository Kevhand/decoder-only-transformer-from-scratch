"""Tests for Decoder_Blocks/decoder_block.py. Run from the project root:  python -m tests.test_decoder_block"""

from contextlib import contextmanager

import torch

from Attention_Mechanism.multi_head_causal_attention import MultiHeadCausalAttention
from Decoder_Blocks.decoder_block import DecoderBlock

torch.manual_seed(42)
D, H, FF, MAX, T = 64, 4, 256, 32, 12


@contextmanager
def sdpa(enabled):
    old = MultiHeadCausalAttention.use_sdpa
    MultiHeadCausalAttention.use_sdpa = enabled
    try:
        yield
    finally:
        MultiHeadCausalAttention.use_sdpa = old


def make_block():
    block = DecoderBlock(D, H, FF, MAX)
    block.eval()
    return block


def test_shape_and_gradients():
    block = make_block()
    x = torch.randn(2, T, D, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    (out * torch.randn_like(out)).sum().backward()
    missing = [n for n, p in block.named_parameters() if p.grad is None or torch.count_nonzero(p.grad) == 0]
    assert not missing, f"no gradient for {missing}"
    print("PASS  output shape matches input; every parameter receives a gradient")


def test_pre_ln_residuals():
    # With both sublayer output projections zeroed, a pre-LN block is exactly the identity:
    # the residual stream carries the un-normalized input straight through.
    block = make_block()
    with torch.no_grad():
        for linear in (block.attention.W_o, block.ffn.linear2):
            linear.weight.zero_()
            linear.bias.zero_()
        x = torch.randn(2, T, D) * 3 + 1
        assert torch.equal(block(x), x), "residual path is not the raw input (pre-LN wiring)"
    print("PASS  pre-LN wiring: with sublayer outputs zeroed, the block returns its input exactly")


def test_cache_equivalence():
    block = make_block()
    x = torch.randn(2, T, D)
    for enabled in (False, True):
        with sdpa(enabled), torch.no_grad():
            full = block(x)
            for chunks in ([(i, i + 1) for i in range(T)], [(0, 3), (3, 7), (7, 12)]):
                cache, outs = None, []
                for s, e in chunks:
                    out, cache = block(x[:, s:e], past_kv=cache, use_cache=True)
                    outs.append(out)
                assert torch.allclose(full, torch.cat(outs, 1), atol=1e-5)
                assert cache[0].shape == (2, H, T, D // H) and cache[1].shape == cache[0].shape
    print("PASS  token-by-token and chunked caching match a full pass (manual and fused attention)")


def test_causality():
    block = make_block()
    x = torch.randn(1, T, D)
    changed = x.clone()
    # A random change, not a constant: adding the same value to every feature is
    # removed by LayerNorm, so later positions would see no difference at all.
    changed[:, 8] += torch.randn(D) * 3
    for enabled in (False, True):
        with sdpa(enabled), torch.no_grad():
            a, b = block(x), block(changed)
        assert torch.equal(a[:, :8], b[:, :8]), "a future token changed an earlier output"
        for pos in range(8, T):
            assert not torch.allclose(a[:, pos], b[:, pos]), f"position {pos} should see the change"
    print("PASS  causality: positions 0..7 unchanged, positions 8..11 each changed (both attention paths)")


def test_bad_cache_rejected():
    block = make_block()
    x = torch.randn(2, 1, D)
    good_k = torch.randn(2, H, 5, D // H)
    bad = {
        "K/V shapes differ": (good_k, torch.randn(2, H, 4, D // H)),
        "wrong batch": (torch.randn(3, H, 5, D // H),) * 2,
        "wrong heads": (torch.randn(2, H + 1, 5, D // H),) * 2,
        "wrong head_dim": (torch.randn(2, H, 5, D // H + 2),) * 2,
    }
    for name, cache in bad.items():
        try:
            block(x, past_kv=cache, use_cache=True)
        except AssertionError:
            continue
        raise AssertionError(f"cache with {name} was not rejected")
    print("PASS  caches with mismatched K/V, batch, heads or head_dim are rejected")


def test_mixed_precision_with_cache():
    block = make_block()
    x = torch.randn(2, T, D)
    with torch.no_grad():
        reference = block(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            full = block(x)
            cache, outs = None, []
            for s, e in [(0, 5), (5, 6), (6, 12)]:
                out, cache = block(x[:, s:e], past_kv=cache, use_cache=True)
                outs.append(out)
    chunked = torch.cat(outs, 1)
    assert torch.isfinite(full).all() and torch.isfinite(chunked).all()
    assert (full.float() - reference).abs().max() < 0.1, "bf16 result far from float32"
    assert (full.float() - chunked.float()).abs().max() < 0.1, "bf16 cached result far from bf16 full pass"
    print("PASS  bf16 autocast works with and without the cache, close to the float32 result")


if __name__ == "__main__":
    test_shape_and_gradients()
    test_pre_ln_residuals()
    test_cache_equivalence()
    test_causality()
    test_bad_cache_rejected()
    test_mixed_precision_with_cache()
    print("\nAll decoder block tests passed.")