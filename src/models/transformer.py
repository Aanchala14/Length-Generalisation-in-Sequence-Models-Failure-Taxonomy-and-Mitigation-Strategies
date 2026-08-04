import torch
import torch.nn as nn

from .positional_encoding import (
    LearnedPositionalEncoding,
    NoPositionalEncoding,
    SinusoidalPositionalEncoding
)

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
        max_length=20,
        positional_encoding="learned"
    ):

        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        # Positional embeddings
        if positional_encoding == "learned":
            self.position_embedding = LearnedPositionalEncoding(
                max_length=max_length,
                embedding_dim=embedding_dim
                )

        elif positional_encoding == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEncoding(
                max_length=max_length,
                embedding_dim=embedding_dim
            )

        elif positional_encoding in ["none", "nope"]:
            self.position_embedding = NoPositionalEncoding()
        else:
            raise ValueError(
                f"Unknown positional encoding: {positional_encoding}"
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