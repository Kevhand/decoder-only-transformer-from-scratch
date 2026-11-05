"""Tests for Embedding/embeddings.py. Run from the project root:  python -m tests.test_embeddings"""

import torch

from Embedding.embeddings import TokenEmbedding, QKV_Projection

torch.manual_seed(0)


def test_token_embedding():
    vocab, d_model, pad = 50, 16, 7
    emb = TokenEmbedding(vocab, d_model, pad)

    ids = torch.tensor([[1, 2, pad, 3], [pad, pad, 4, 5]])
    out = emb(ids)
    assert out.shape == (2, 4, d_model)

    # Output is exactly the matching row of the weight matrix.
    assert torch.equal(out[0, 1], emb.embedding.weight[2])

    # The PAD row starts at zero, and lookups give it no gradient.
    assert torch.count_nonzero(emb.embedding.weight[pad]) == 0
    out.sum().backward()
    grad = emb.embedding.weight.grad
    assert torch.count_nonzero(grad[pad]) == 0, "PAD row received a gradient from lookups"
    assert torch.count_nonzero(grad[1]) > 0, "a used token received no gradient"
    assert torch.count_nonzero(grad[10]) == 0, "an unused token received a gradient"
    print("PASS  TokenEmbedding: shape, row lookup, PAD row zero with no lookup gradient")


def test_qkv_shapes_and_head_split():
    batch, seq, d_model, heads = 2, 5, 32, 4
    head_dim = d_model // heads
    qkv = QKV_Projection(d_model, heads)
    x = torch.randn(batch, seq, d_model)

    Q, K, V = qkv(x)
    for t in (Q, K, V):
        assert t.shape == (batch, heads, seq, head_dim)

    # Hand computation: head h holds columns h*head_dim .. (h+1)*head_dim of the projection.
    with torch.no_grad():
        for proj, out in ((qkv.q_proj, Q), (qkv.k_proj, K), (qkv.v_proj, V)):
            full = x @ proj.weight.T + proj.bias
            for h in range(heads):
                expected = full[:, :, h * head_dim:(h + 1) * head_dim]
                assert torch.allclose(out[:, h], expected, atol=1e-6)
    print("PASS  QKV_Projection: shapes (batch, heads, seq, head_dim) and head split match hand computation")


def test_qkv_positions_independent():
    qkv = QKV_Projection(16, 4)
    x = torch.randn(1, 6, 16)
    changed = x.clone()
    changed[0, 3] += 5.0

    with torch.no_grad():
        a = qkv(x)
        b = qkv(changed)
    for ta, tb in zip(a, b):
        others = [0, 1, 2, 4, 5]
        assert torch.equal(ta[:, :, others], tb[:, :, others]), "changing one token affected another"
        assert not torch.allclose(ta[:, :, 3], tb[:, :, 3]), "changed token's projection did not change"
    print("PASS  QKV_Projection: each position projected independently (with negative control)")


def test_qkv_rejects_bad_config():
    try:
        QKV_Projection(30, 4)
    except AssertionError:
        print("PASS  QKV_Projection: d_model not divisible by num_heads is rejected")
        return
    raise AssertionError("expected an error for d_model=30, num_heads=4")


if __name__ == "__main__":
    test_token_embedding()
    test_qkv_shapes_and_head_split()
    test_qkv_positions_independent()
    test_qkv_rejects_bad_config()
    print("\nAll embedding tests passed.")