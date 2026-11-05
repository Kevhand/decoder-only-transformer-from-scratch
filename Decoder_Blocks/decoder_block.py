import torch
import torch.nn as nn
from LayerNorm.layer_norm import LayerNorm
from Embedding.embeddings import QKV_Projection
from Positional_Encoding.rotary_positional_encoding import ROPE
from Attention_Mechanism.multi_head_causal_attention import MultiHeadCausalAttention
from FeedForward_Network.ffn import FFN

class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        eps=1e-5
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.ln1 = LayerNorm(d_model, eps)
        self.qkv = QKV_Projection(d_model, num_heads)
        self.rope = ROPE(d_model // num_heads, max_seq_len)
        self.attention = MultiHeadCausalAttention(
            d_model, max_seq_len
        )

        self.ln2 = LayerNorm(d_model, eps)
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, past_kv=None, use_cache=False):
        # past_kv: optional tuple (past_K, past_V)
        # Each cache has shape (B, heads, past_len, head_dim)

        if past_kv is None:
            past_len = 0
        else:
            past_k, past_v = past_kv
            past_len = past_k.size(2)

            assert past_k.shape == past_v.shape
            assert past_k.size(0) == x.size(0)
            assert past_k.size(1) == self.qkv.num_heads
            assert past_k.size(3) == self.qkv.head_dim

        # ----- Attention sublayer -----
        residual = x
        normalized_x = self.ln1(x)

        Q, K, V = self.qkv(normalized_x)

        # Rotate only the NEW Q and K at their absolute positions.
        Q, K = self.rope(
            Q, K,
            q_offset=past_len,
            k_offset=past_len
        )

        # Append new, already-rotated keys and new values.
        if past_kv is not None:
            K_all = torch.cat([past_k, K], dim=2)
            V_all = torch.cat([past_v, V], dim=2)
        else:
            K_all = K
            V_all = V

        # Attention sees all previous keys and current-chunk keys.
        attention_output, attention_weights = self.attention(
            Q, K_all, V_all, past_len=past_len
        )

        x = residual + attention_output

        # ----- Feed-forward sublayer -----
        residual = x
        ffn_output = self.ffn(self.ln2(x))
        x = residual + ffn_output

        if use_cache:
            return x, (K_all, V_all)

        return x