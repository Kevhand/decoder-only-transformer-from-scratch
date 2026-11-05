import torch  
import torch.nn as nn

from Embedding.embeddings import TokenEmbedding, QKV_Projection
from Attention_Mechanism.multi_head_causal_attention import MultiHeadCausalAttention
from Positional_Encoding.rotary_positional_encoding import ROPE
from FeedForward_Network.ffn import FFN
from LayerNorm.layer_norm import LayerNorm
from .decoder_block import DecoderBlock



class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, padding_id, num_heads, d_ff, max_seq_len, num_layers):
        super().__init__()
        
        self.token_embedding = TokenEmbedding(vocab_size, d_model, padding_id)
        self.padding_id = padding_id

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])


        self.final_layer_norm = LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.token_embedding.embedding.weight
        self._initialize_weights()


    def _initialize_weights(self):
        # Initialize Linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # After general Linear initialization
        # Scale residual projections
        residual_std = 0.02 / (2 * len(self.blocks)) ** 0.5

        for block in self.blocks:
            # Attention output projection
            nn.init.normal_(
                block.attention.W_o.weight,
                mean=0.0,
                std=residual_std
            )

            # Second FFN linear layer
            nn.init.normal_(
                block.ffn.linear2.weight,
                mean=0.0,
                std=residual_std
            )
        # Initialize custom LayerNorm layers
        for module in self.modules():
            if isinstance(module, LayerNorm):
                nn.init.ones_(module.gamma)
                nn.init.zeros_(module.beta)

        # Initialize embeddings LAST, so tied weights
        # receive the embedding initialization
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()

    def forward(self, token_ids, past_kv=None, use_cache=False):
        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        elif len(past_kv) != len(self.blocks):
            raise ValueError(
                f"Length of past_kv ({len(past_kv)}) does not match number of blocks ({len(self.blocks)})."
            )

        x = self.token_embedding(token_ids)

        new_kvs = []

        for block, past_kv in zip(self.blocks, past_kv):
            if use_cache:
                x, new_kv = block(x, past_kv=past_kv, use_cache=True)
                new_kvs.append(new_kv)
            else:
                x = block(x, past_kv=past_kv, use_cache=False)

        x = self.final_layer_norm(x)

        logits = self.lm_head(x)

        pad_mask = torch.zeros(
            logits.size(-1), dtype=torch.bool, device=logits.device
        )
        pad_mask[self.padding_id] = True

        logits = logits.masked_fill(pad_mask, float("-inf"))

        if use_cache:
            return logits, new_kvs


        return logits
