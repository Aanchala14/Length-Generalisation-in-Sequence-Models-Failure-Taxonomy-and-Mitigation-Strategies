import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from src.data.time_series_dataset import TimeSeriesPrefixDataset
from src.models.time_series_transformer import TimeSeriesTransformer
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    dataset_name = config["dataset"]
    train_length = config["train_length"]
    test_lengths = config["test_lengths"]
    data_root = config.get("data_root", "data/realworld")

    seed = config.get("seed", 42)
    positional_encoding = config.get("positional_encoding", "learned")
    experiment_name = config.get(
        "experiment_name",
        f"{positional_encoding}_seed{seed}",
    )

    reference_dataset = TimeSeriesPrefixDataset(
        dataset_name=dataset_name,
        data_root=data_root,
        split="test",
        length=train_length,
    )

    model = TimeSeriesTransformer(
        input_dim=reference_dataset.input_dim,
        num_classes=reference_dataset.num_classes,
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        positional_encoding=positional_encoding,
    )

    checkpoint_dir = config.get(
        "checkpoint_dir",
        "outputs/checkpoints/realworld",
    )

    safe_dataset = dataset_name.lower()
    checkpoint_path = (
        f"{checkpoint_dir}/"
        f"{safe_dataset}_train{train_length}_{experiment_name}.pt"
    )

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )

    model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")

    rows = []

    for length in test_lengths:
        test_dataset = TimeSeriesPrefixDataset(
            dataset_name=dataset_name,
            data_root=data_root,
            split="test",
            length=length,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                predictions = logits.argmax(dim=-1)

                correct += (predictions == y).sum().item()
                total += y.size(0)

        accuracy = 100 * correct / total

        print(f"Length {length} | Accuracy: {accuracy:.2f}%")

        rows.append([
            dataset_name,
            train_length,
            length,
            positional_encoding,
            seed,
            accuracy,
        ])

    results_dir = config.get(
        "results_dir",
        "outputs/results/realworld",
    )
    os.makedirs(results_dir, exist_ok=True)

    results_path = (
        f"{results_dir}/"
        f"{safe_dataset}_train{train_length}_{experiment_name}_results.csv"
    )

    with open(results_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Dataset",
            "Train Length",
            "Test Length",
            "Positional Encoding",
            "Seed",
            "Accuracy",
        ])
        writer.writerows(rows)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()