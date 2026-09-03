import math

import torch
import torch.nn as nn


def get_alibi_slopes(num_heads):
    """
    Create ALiBi slopes for each attention head
    this follows the standard ALiBi slope construction
    """

    def get_slopes_power_of_2(n):
        start = 2 ** (
            -2 ** -(math.log2(n) - 3)
        )

        ratio = start

        return [
            start * ratio ** i
            for i in range(n)
        ]

    if math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)

    else:
        closest_power_of_2 = 2 ** math.floor(
            math.log2(num_heads)
        )

        slopes = get_slopes_power_of_2(
            closest_power_of_2
        )

        extra_slopes = get_slopes_power_of_2(
            2 * closest_power_of_2
        )

        slopes.extend(
            extra_slopes[0::2][
                : num_heads - closest_power_of_2
            ]
        )

    return torch.tensor(
        slopes,
        dtype=torch.float32
    )


def build_alibi_bias(num_heads, sequence_length, device):

    slopes = get_alibi_slopes(num_heads).to(device)

    positions = torch.arange(
        sequence_length,
        device=device
    )

    distances = (
        positions[None, :]
        - positions[:, None]
    ).abs()

    bias = -distances.float()

    bias = bias.unsqueeze(0).unsqueeze(0)

    slopes = slopes.view(
        1,
        num_heads,
        1,
        1
    )

    return slopes * bias

def apply_rope(x):

    batch_size, num_heads, sequence_length, head_dim = x.shape

    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")

    positions = torch.arange(
        sequence_length,
        device=x.device
    ).float()

    inv_freq = 1.0 / (
        10000 ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=x.device
            ).float() / head_dim
        )
    )

    freqs = torch.outer(
        positions,
        inv_freq
    )

    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    return torch.stack(
        (rotated_even, rotated_odd),
        dim=-1
    ).flatten(-2)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional ALiBi bias.
    """

    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout=0.1,
        use_alibi=False,
        use_rope=False
    ):
        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.use_alibi = use_alibi
        self.use_rope = use_rope
        self.q_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.k_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.v_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.out_proj = nn.Linear(
            embedding_dim,
            embedding_dim
        )

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        batch_size, sequence_length, _ = x.shape

        x = x.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, sequence_length, _ = x.shape

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

    def forward(self, x, return_attention=False):
        batch_size, sequence_length, _ = x.shape

        q = self.split_heads(
            self.q_proj(x)
        )

        k = self.split_heads(
            self.k_proj(x)
        )

        v = self.split_heads(
            self.v_proj(x)
        )

        if self.use_rope:
            q = apply_rope(q)
            k = apply_rope(k)

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1)
        )

        attention_scores = attention_scores / math.sqrt(
            self.head_dim
        )

        if self.use_alibi:
            attention_scores = attention_scores + build_alibi_bias(
                num_heads=self.num_heads,
                sequence_length=sequence_length,
                device=x.device
            )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        context = torch.matmul(
            attention_weights,
            v
        )

        context = self.combine_heads(context)

        output = self.out_proj(context)
        if return_attention:
            return output, attention_weights

        return output


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads,
        feedforward_dim,
        dropout=0.1,
        use_alibi=False,
        use_rope=False
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_alibi=use_alibi,
            use_rope=use_rope
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(
                embedding_dim,
                feedforward_dim
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                feedforward_dim,
                embedding_dim
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        if return_attention:
            attention_output, attention_weights = self.attention(
                x,
                return_attention=True
            )
        else:
            attention_output = self.attention(x)
            attention_weights = None

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        feedforward_output = self.feedforward(x)

        x = self.norm2(
            x + self.dropout(feedforward_output)
        )

        if return_attention:
            return x, attention_weights
        return x