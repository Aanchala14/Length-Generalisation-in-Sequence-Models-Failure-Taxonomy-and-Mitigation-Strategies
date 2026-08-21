import os
import argparse
import random

import torch

from src.data.dataloader import get_dataloader
from src.models.transformer import TransformerModel
from src.utils.config import load_config
from src.utils.seed import set_seed


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
    seed = config.get("seed", 42)
    set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    task = config["task"]
    train_length = config["train_length"]

    positional_encoding = config.get(
        "positional_encoding",
        "learned"
    )

    experiment_name = config.get(
        "experiment_name",
        f"{positional_encoding}_seed{seed}"
    )

    checkpoint_dir = config.get(
        "checkpoint_dir",
        "outputs/checkpoints"
    )

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

    data_dir = config.get(
        "data_dir",
        f"data/synthetic/{task}"
    )

    train_lengths = config.get("train_lengths")

    if train_lengths:
        train_loaders = [
            get_dataloader(
                f"{data_dir}/train_{length}.jsonl",
                batch_size=config["batch_size"],
                shuffle=True
            )
            for length in train_lengths
        ]

    else:
        train_loaders = [
            get_dataloader(
                f"{data_dir}/train.jsonl",
                batch_size=config["batch_size"],
                shuffle=True
            )
        ]

    model = TransformerModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_length=max_length,
        positional_encoding=positional_encoding
    ).to(device)

    initial_checkpoint = config.get("initial_checkpoint")

    if initial_checkpoint:
        model.load_state_dict(
            torch.load(
                initial_checkpoint,
                map_location=device
            )
        )

        print(f"Loaded initial checkpoint from: {initial_checkpoint}")

    pad_token = config.get("pad_token")

    criterion = (
        torch.nn.CrossEntropyLoss(ignore_index=pad_token)
        if pad_token is not None
        else torch.nn.CrossEntropyLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    train_losses = []

    loader_indices = list(range(len(train_loaders)))

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0

        random.shuffle(loader_indices)

        for loader_index in loader_indices:
            train_loader = train_loaders[loader_index]

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

        total_batches = sum(
            len(train_loader)
            for train_loader in train_loaders
        )

        average_loss = epoch_loss / total_batches
        train_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"Loss: {average_loss:.4f}"
        )

    print("\nTraining finished!")
    print("Training losses:")
    print(train_losses)

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = (
        f"{checkpoint_dir}/"
        f"{task}_train{train_length}_{experiment_name}.pt"
    )

    torch.save(
        model.state_dict(),
        checkpoint_path
    )

    print(f"\nModel saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()