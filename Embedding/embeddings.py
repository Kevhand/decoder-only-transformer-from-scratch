import torch
import torch.nn as nn



class TokenEmbedding(nn.Module):
    def  __init__(self, vocab_size, d_model, padding_id):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model

        """
        by giving padding_idx = padding_id, the embedding layer will not update the weights for the padding token during training. 
        This is useful when you have a special token in your vocabulary that represents padding and you don't want it to contribute to the learning process.
        """
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_id) 

    def forward(self, token_ids):
        return self.embedding(token_ids)



class QKV_Projection(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        assert d_model % num_heads == 0, \
            "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)



        self.head_dim = d_model // num_heads

    def forward(self, x):
        batch, seq_len, _ = x.shape

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)


        Q = Q.view(batch, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch, seq_len, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        return Q, K, V




