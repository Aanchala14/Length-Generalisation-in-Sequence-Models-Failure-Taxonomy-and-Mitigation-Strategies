import torch

from src.models.positional_encoding import LearnedPositionalEncoding


embedding_dim = 128
max_length = 20

position_encoding = LearnedPositionalEncoding(
    max_length=max_length,
    embedding_dim=embedding_dim
)

x = torch.randn(4, 20, embedding_dim)

output = position_encoding(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)