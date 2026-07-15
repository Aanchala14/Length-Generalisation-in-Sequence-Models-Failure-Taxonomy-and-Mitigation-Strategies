import torch

from src.models.transformer import TransformerModel


model = TransformerModel()

x = torch.randint(
    low=0,
    high=100,
    size=(4, 20)
)

output = model(x)

print("Input shape :", x.shape)
print("Output shape:", output.shape)