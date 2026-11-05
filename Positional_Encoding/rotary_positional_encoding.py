import torch
import torch.nn as nn

class ROPE(nn.Module):
    def __init__(self, head_dim, max_seq_len):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

        assert head_dim % 2 == 0 # head_dim must be even for ROPE to work

        # Generating freqiencies
        i = torch.arange(0, head_dim, 2)

        frequencies = 1.0 / 10000 ** (i / head_dim)

        # Generating positions
        positions = torch.arange(max_seq_len)

        # Generating Angles

        theta = positions[:, None] * frequencies[None, :] # shape: (max_seq_len, head_dim/2) for every sequence position, calculate its roatation angle.

        self.register_buffer('cos_theta', torch.cos(theta), persistent=False)
        self.register_buffer('sin_theta', torch.sin(theta), persistent=False)

    def forward(self, q, k, q_offset=0, k_offset=0):
        q_seq_length = q.size(2)
        k_seq_length = k.size(2)


        if q_offset < 0 or k_offset < 0:
            raise ValueError("Offsets must be non-negative integers.")

        q_end = q_offset + q_seq_length
        k_end = k_offset + k_seq_length


        if q_end > self.max_seq_len:
            raise ValueError(f"Q position range [{q_offset}, {q_end - 1}] "
            f"exceeds max_seq_length={self.max_seq_len}"
        )

        if k_end > self.max_seq_len:
            raise ValueError(f"K position range [{k_offset}, {k_end - 1}] "
            f"exceeds max_seq_length={self.max_seq_len}"
        )

        q_cos = self.cos_theta[q_offset : q_end]
        q_sin = self.sin_theta[q_offset : q_end]

        k_cos = self.cos_theta[k_offset : k_end]
        k_sin = self.sin_theta[k_offset : k_end]


        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]

        k_even = k[..., 0::2]
        k_odd = k[..., 1::2]


        q_rotated_even = q_even * q_cos - q_odd * q_sin
        q_rotated_odd = q_even * q_sin + q_odd * q_cos

        k_rotated_even = k_even * k_cos - k_odd * k_sin
        k_rotated_odd = k_even * k_sin + k_odd * k_cos


        q_rotated = torch.stack([q_rotated_even, q_rotated_odd], dim=-1).flatten(-2)

        k_rotated = torch.stack([k_rotated_even, k_rotated_odd], dim=-1).flatten(-2)

        return q_rotated, k_rotated


