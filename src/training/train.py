import torch

from src.models.transformer import TransformerModel
from src.data.dataloader import get_dataloader
from src.utils.config import load_config

config = load_config("configs/baseline.yaml")

# -----------------------------
# Device
# -----------------------------
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# -----------------------------
# Data
# -----------------------------
train_loader = get_dataloader(
    f"data/synthetic/{config['task']}/train.jsonl",
    batch_size=config["batch_size"],
    shuffle=True
)


# -----------------------------
# Model
# -----------------------------
model = TransformerModel().to(device)

PAD_TOKEN = None

criterion = torch.nn.CrossEntropyLoss(
    ignore_index=PAD_TOKEN
) if PAD_TOKEN is not None else torch.nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3
)


NUM_EPOCHS = 10

train_losses = []

for epoch in range(NUM_EPOCHS):

    model.train()

    epoch_loss = 0.0

    for x, y in train_loader:

        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            y.view(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(train_loader)

    train_losses.append(average_loss)

    print(
        f"Epoch {epoch+1}/{NUM_EPOCHS} | "
        f"Loss: {average_loss:.4f}"
    )

print("\nTraining finished!")

print("Training losses:")

print(train_losses)

import os

os.makedirs("outputs/checkpoints", exist_ok=True)

torch.save(
    model.state_dict(),
    "outputs/checkpoints/copy_baseline.pt"
)

print("\nModel saved successfully!")