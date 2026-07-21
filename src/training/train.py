import os
import argparse
import torch

from src.data.dataloader import get_dataloader
from src.models.transformer import TransformerModel
from src.utils.config import load_config


DEFAULT_CONFIG_PATH = "configs/copy_baseline.yaml"


def get_sequence_length(task, length):
    if task == "addition":
        return (2 * length) + 1

    return length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH
    )
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    task = config["task"]
    train_length = config["train_length"]

    max_train_sequence_length = get_sequence_length(
        task,
        train_length
    )

    max_test_sequence_length = max(
        get_sequence_length(task, length)
        for length in config.get("test_lengths", [train_length])
    )

    max_length = config.get(
        "max_length",
        max(max_train_sequence_length, max_test_sequence_length)
    )

    train_loader = get_dataloader(
        f"data/synthetic/{task}/train.jsonl",
        batch_size=config["batch_size"],
        shuffle=True
    )

    model = TransformerModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_length=max_length
    ).to(device)

    pad_token = config.get("pad_token")

    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_token
    ) if pad_token is not None else torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    train_losses = []

    for epoch in range(config["epochs"]):
        model.train()

        epoch_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            loss = criterion(
                outputs.reshape(-1, outputs.size(-1)),
                y.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        average_loss = epoch_loss / len(train_loader)
        train_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"Loss: {average_loss:.4f}"
        )

    print("\nTraining finished!")
    print("Training losses:")
    print(train_losses)

    os.makedirs("outputs/checkpoints", exist_ok=True)

    checkpoint_path = (
        f"outputs/checkpoints/"
        f"{task}_train{train_length}_baseline.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(f"\nModel saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()