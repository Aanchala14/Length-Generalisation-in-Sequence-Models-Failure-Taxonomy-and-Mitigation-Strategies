import torch.nn as nn

from src.models.attention import TransformerBlock
from src.models.positional_encoding import (
    LearnedPositionalEncoding,
    NoPositionalEncoding,
    SinusoidalPositionalEncoding,
)


class TimeSeriesTransformer(nn.Module):
    def __init__(
        self,
        input_dim,
        num_classes,
        embedding_dim=128,
        num_heads=4,
        num_layers=2,
        feedforward_dim=256,
        dropout=0.1,
        max_length=1152,
        positional_encoding="learned",
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, embedding_dim)

        if positional_encoding == "learned":
            self.position_embedding = LearnedPositionalEncoding(
                max_length=max_length,
                embedding_dim=embedding_dim,
            )
            use_alibi = False
            use_rope = False

        elif positional_encoding == "sinusoidal":
            self.position_embedding = SinusoidalPositionalEncoding(
                max_length=max_length,
                embedding_dim=embedding_dim,
            )
            use_alibi = False
            use_rope = False

        elif positional_encoding in ["none", "nope"]:
            self.position_embedding = NoPositionalEncoding()
            use_alibi = False
            use_rope = False

        elif positional_encoding == "alibi":
            self.position_embedding = NoPositionalEncoding()
            use_alibi = True
            use_rope = False

        elif positional_encoding == "rope":
            self.position_embedding = NoPositionalEncoding()
            use_alibi = False
            use_rope = True

        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")

        self.layers = nn.ModuleList([
            TransformerBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                feedforward_dim=feedforward_dim,
                dropout=dropout,
                use_alibi=use_alibi,
                use_rope=use_rope,
            )
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.position_embedding(x)

        for layer in self.layers:
            x = layer(x)

        pooled = x.mean(dim=1)

        return self.classifier(pooled)