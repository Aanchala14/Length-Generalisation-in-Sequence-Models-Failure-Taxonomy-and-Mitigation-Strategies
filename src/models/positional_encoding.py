import torch
import torch.nn as nn


class LearnedPositionalEncoding(nn.Module):
    """
    Learnable absolute positional embeddings.
    """

    def __init__(self, max_length, embedding_dim):
        super().__init__()

        self.position_embedding = nn.Embedding(
            max_length,
            embedding_dim
        )

    def forward(self, x):
        """
        x shape:
        (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length, _ = x.shape

        positions = torch.arange(
            sequence_length,
            device=x.device
        )

        positions = positions.unsqueeze(0).expand(
            batch_size,
            sequence_length
        )

        return x + self.position_embedding(positions)