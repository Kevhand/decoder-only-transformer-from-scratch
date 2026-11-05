import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadCausalAttention(nn.Module):
    # Fused attention (PyTorch's scaled_dot_product_attention): same result, much
    # less memory, faster. Set to False to use the manual path and get the
    # attention weights back, e.g. for interpretability.
    use_sdpa = True

    def __init__(self, d_model, max_seq_len):
        super().__init__()
        self.d_model = d_model
        self.W_o = nn.Linear(d_model, d_model, bias=True)

    def forward(self, Q, K, V, past_len=0):
        batch, num_heads, seq_len, head_dim = Q.shape
        k_len = K.size(-2)
        if k_len != past_len + seq_len:
            raise ValueError(f"Expected {past_len + seq_len} keys (past_len + seq_len), got {k_len}")

        query_position = torch.arange(past_len, past_len + seq_len, device=Q.device)
        key_position = torch.arange(k_len, device=K.device)

        if self.use_sdpa:
            if past_len == 0:
                # Square causal mask: lets PyTorch pick its fastest kernel.
                context = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            else:
                mask = key_position.unsqueeze(0) <= query_position.unsqueeze(1)  # True = may attend
                context = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
            attention_weights = None
        else:
            scores = Q @ K.transpose(-2, -1) / math.sqrt(head_dim)
            mask = key_position.unsqueeze(0) <= query_position.unsqueeze(1)
            scores = scores.masked_fill(~mask, float("-inf"))
            attention_weights = torch.softmax(scores, dim=-1)
            context = attention_weights @ V

        context = context.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(context), attention_weights