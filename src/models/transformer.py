import torch
import torch.nn as nn

from .positional_encoding import LearnedPositionalEncoding


class TransformerModel(nn.Module):
    """
    Baseline Transformer model for synthetic sequence tasks.
    """

    def __init__(
        self,
        vocab_size=100,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_length=20
    ):

        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Positional embeddings
        self.position_embedding = LearnedPositionalEncoding(
            max_length=max_length,
            embedding_dim=embedding_dim
        )

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Output projection
        self.output_layer = nn.Linear(
            embedding_dim,
            vocab_size
        )

    def forward(self, x):

        # x shape:
        # (batch_size, sequence_length)

        x = self.token_embedding(x)

        x = self.position_embedding(x)

        x = self.encoder(x)

        logits = self.output_layer(x)

        return logits