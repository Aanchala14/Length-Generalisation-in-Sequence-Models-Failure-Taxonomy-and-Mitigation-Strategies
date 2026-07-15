import torch

print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps")

x = torch.rand(3, 3).to(device)

print(x)