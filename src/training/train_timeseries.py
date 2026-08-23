import argparse
import os

import torch
from torch.utils.data import DataLoader

from src.data.time_series_dataset import TimeSeriesPrefixDataset
from src.models.time_series_transformer import TimeSeriesTransformer
from src.utils.config import load_config
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
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

    dataset_name = config["dataset"]
    train_length = config["train_length"]
    data_root = config.get("data_root", "data/realworld")

    train_dataset = TimeSeriesPrefixDataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split="train",
        length=train_length,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    positional_encoding = config.get("positional_encoding", "learned")
    experiment_name = config.get(
        "experiment_name",
        f"{positional_encoding}_seed{seed}",
    )

    model = TimeSeriesTransformer(
        input_dim=train_dataset.input_dim,
        num_classes=train_dataset.num_classes,
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        positional_encoding=positional_encoding,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
    )

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = logits.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(
            f"Epoch {epoch + 1}/{config['epochs']} | "
            f"Loss: {avg_loss:.4f} | Train Acc: {accuracy:.2f}%"
        )

    checkpoint_dir = config.get(
        "checkpoint_dir",
        "outputs/checkpoints/realworld",
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    safe_dataset = dataset_name.lower()
    checkpoint_path = (
        f"{checkpoint_dir}/"
        f"{safe_dataset}_train{train_length}_{experiment_name}.pt"
    )

    torch.save(model.state_dict(), checkpoint_path)

    print(f"\nModel saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()