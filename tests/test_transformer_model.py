"""Tests for Decoder_Blocks/transformer_model.py. Run from the project root:  python -m tests.test_transformer_model"""

import io
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

from Attention_Mechanism.multi_head_causal_attention import MultiHeadCausalAttention
from Decoder_Blocks.transformer_model import TransformerModel

V, D, H, FF, LAYERS, MAX, PAD = 1000, 64, 4, 256, 3, 32, 0
T = 12


@contextmanager
def sdpa(enabled):
    old = MultiHeadCausalAttention.use_sdpa
    MultiHeadCausalAttention.use_sdpa = enabled
    try:
        yield
    finally:
        MultiHeadCausalAttention.use_sdpa = old


def make_model(seed=42):
    torch.manual_seed(seed)
    return TransformerModel(vocab_size=V, d_model=D, padding_id=PAD, num_heads=H, d_ff=FF,
                            max_seq_len=MAX, num_layers=LAYERS).eval()


def random_ids(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(1, V, shape, generator=g)  # real tokens only


def real(logits):
    """Logits without the PAD column, which is always -inf."""
    return torch.cat([logits[..., :PAD], logits[..., PAD + 1:]], dim=-1)


def test_output_and_pad_mask():
    model = make_model()
    with torch.no_grad():
        logits = model(random_ids((2, T)))
    assert logits.shape == (2, T, V)
    assert torch.isneginf(logits[..., PAD]).all(), "PAD logit must be -inf everywhere"
    assert torch.isfinite(real(logits)).all()
    print("PASS  logits shaped (B, T, vocab); PAD column -inf, all other logits finite")


def test_initialization():
    model = make_model()
    emb = model.token_embedding.embedding.weight
    residual_std = 0.02 / math.sqrt(2 * LAYERS)
    assert model.lm_head.weight is emb, "LM head must share the embedding tensor"
    assert torch.count_nonzero(emb[PAD]) == 0
    assert abs(emb[1:].std().item() - 0.02) < 0.002
    for block in model.blocks:
        assert abs(block.attention.W_o.weight.std().item() - residual_std) < 0.002
        assert abs(block.ffn.linear2.weight.std().item() - residual_std) < 0.002
        assert abs(block.ffn.linear1.weight.std().item() - 0.02) < 0.002
    assert all(torch.count_nonzero(p) == 0 for n, p in model.named_parameters() if n.endswith(".bias"))
    assert not any("mask" in k for k in model.state_dict()), "no leftover mask buffers in checkpoints"
    print(f"PASS  init: tied weights, PAD row 0, embedding std 0.02, residual projections std "
          f"{residual_std:.4f}, biases 0, no mask buffers")


def test_initial_loss():
    model = make_model()
    ids = random_ids((4, T))
    targets = random_ids((4, T), seed=1)
    with torch.no_grad():
        logits = model(ids)
    loss = F.cross_entropy(logits.view(-1, V), targets.view(-1), ignore_index=PAD).item()
    baseline = math.log(V - 1)  # PAD can never be predicted
    std = real(logits).std().item()
    assert abs(loss - baseline) < 0.1, f"initial loss {loss:.3f} far from ln(V-1) = {baseline:.3f}"
    assert std < 0.5, f"initial logits too large (std {std:.3f})"
    print(f"PASS  initial loss {loss:.3f} ~ ln(vocab-1) = {baseline:.3f}; logit std {std:.3f}")


def test_final_layer_norm():
    model = make_model()
    ids = random_ids((2, T))
    with torch.no_grad():
        h = model.token_embedding(ids)
        for block in model.blocks:
            h = block(h)
        normed = model.final_layer_norm(h)
        manual = model.lm_head(normed).masked_fill(
            torch.arange(V) == PAD, float("-inf"))
        logits = model(ids)
    var_in = h.var(-1, unbiased=False)
    eps = model.final_layer_norm.eps
    assert torch.allclose(normed.mean(-1), torch.zeros_like(var_in), atol=1e-5)
    assert torch.allclose(normed.var(-1, unbiased=False), var_in / (var_in + eps), atol=1e-4)
    assert torch.allclose(logits, manual)
    print("PASS  final LayerNorm applied before the head (variance matches var/(var+eps)); forward = manual")


def test_cache_equivalence():
    model = make_model()
    ids = random_ids((2, T))
    for enabled in (False, True):
        with sdpa(enabled), torch.no_grad():
            full = real(model(ids))
            for chunks in ([(i, i + 1) for i in range(T)], [(0, 3), (3, 7), (7, 12)]):
                cache, outs = None, []
                for s, e in chunks:
                    out, cache = model(ids[:, s:e], past_kv=cache, use_cache=True)
                    outs.append(real(out))
                diff = (full - torch.cat(outs, 1)).abs().max().item()
                assert diff < 1e-5, f"cached logits differ by {diff}"
                assert len(cache) == LAYERS and all(k.size(2) == T for k, v in cache)
    print("PASS  token-by-token and chunked generation match a full pass across all layers (both attention paths)")


def test_causality():
    model = make_model()
    ids = random_ids((1, T))
    changed = ids.clone()
    changed[0, 8] = (changed[0, 8] % (V - 1)) + 1  # a different real token
    for enabled in (False, True):
        with sdpa(enabled), torch.no_grad():
            a, b = real(model(ids)), real(model(changed))
        assert torch.equal(a[:, :8], b[:, :8]), "a future token changed earlier logits"
        for pos in range(8, T):
            assert not torch.allclose(a[:, pos], b[:, pos]), f"position {pos} should change"
    print("PASS  causality: logits 0..7 unchanged, 8..11 each changed (both attention paths)")


def test_bad_cache_list():
    model = make_model()
    try:
        model(random_ids((2, 1)), past_kv=[None] * (LAYERS - 1), use_cache=True)
    except ValueError:
        print("PASS  a cache list with the wrong number of layers is rejected")
        return
    raise AssertionError("expected a ValueError")


def test_save_load_roundtrip():
    model = make_model(seed=1)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    fresh = make_model(seed=2)
    fresh.load_state_dict(torch.load(buffer))
    assert fresh.lm_head.weight is fresh.token_embedding.embedding.weight, "tying lost after loading"
    ids = random_ids((2, T))
    with torch.no_grad():
        assert torch.equal(model(ids), fresh(ids))
    print("PASS  save/load round trip gives identical logits and keeps the weights tied")


def test_training_steps_keep_pad_row_zero():
    # Same optimizer setup as train.py: AdamW with weight decay on matrices.
    model = make_model().train()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    other = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": other, "weight_decay": 0.0}],
                            lr=1e-3)
    ids = random_ids((4, T + 1))
    ids[:, -3:] = PAD  # include PAD inputs and PAD targets
    x, y = ids[:, :-1], ids[:, 1:]
    for _ in range(5):
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, V), y.reshape(-1), ignore_index=PAD)
        assert torch.isfinite(loss), "PAD targets must be ignored, or the loss is infinite"
        loss.backward()
        opt.step()
    assert torch.count_nonzero(model.token_embedding.embedding.weight[PAD]) == 0
    print("PASS  5 AdamW steps with weight decay and PAD targets: loss finite, PAD row still exactly 0")


def test_overfit_one_batch():
    model = make_model().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ids = random_ids((2, T + 1))
    x, y = ids[:, :-1], ids[:, 1:]
    first = None
    for step in range(150):
        opt.zero_grad()
        loss = F.cross_entropy(model(x).view(-1, V), y.reshape(-1))
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.1, f"could not overfit one batch (loss {loss.item():.3f})"
    print(f"PASS  overfits one batch: loss {first:.2f} -> {loss.item():.4f} in 150 steps")


if __name__ == "__main__":
    test_output_and_pad_mask()
    test_initialization()
    test_initial_loss()
    test_final_layer_norm()
    test_cache_equivalence()
    test_causality()
    test_bad_cache_list()
    test_save_load_roundtrip()
    test_training_steps_keep_pad_row_zero()
    test_overfit_one_batch()
    print("\nAll transformer model tests passed.")