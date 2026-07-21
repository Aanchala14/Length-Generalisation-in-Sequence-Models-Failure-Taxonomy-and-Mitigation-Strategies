'''Load trained model
        ↓
Load test dataset
        ↓
Run inference
        ↓
Compare predictions with targets
        ↓
Compute accuracy
        ↓
Repeat for every sequence length
        ↓
Save results'''

import csv
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


def compute_accuracy(predictions, targets, pad_token=None):
    if pad_token is not None:
        mask = targets != pad_token

        correct = (
            (predictions == targets) & mask
        ).sum().item()

        total = mask.sum().item()

    else:
        correct = (predictions == targets).sum().item()
        total = targets.numel()

    return correct, total


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
    test_lengths = config["test_lengths"]

    max_train_sequence_length = get_sequence_length(
        task,
        train_length
    )

    max_test_sequence_length = max(
        get_sequence_length(task, length)
        for length in test_lengths
    )

    max_length = config.get(
        "max_length",
        max(max_train_sequence_length, max_test_sequence_length)
    )

    model = TransformerModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_length=max_length
    )

    checkpoint_path = (
        f"outputs/checkpoints/"
        f"{task}_train{train_length}_baseline.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    print(f"Loaded model from: {checkpoint_path}")

    pad_token = config.get("pad_token")

    results = []

    for length in test_lengths:
        print(f"\nEvaluating {task} length {length}")

        test_loader = get_dataloader(
            f"data/synthetic/{task}/test_{length}.jsonl",
            batch_size=config["batch_size"],
            shuffle=False
        )

        correct_tokens = 0
        total_tokens = 0
        exact_matches = 0
        total_sequences = 0

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)

                outputs = model(x)
                predictions = outputs.argmax(dim=-1)

                correct, total = compute_accuracy(
                    predictions,
                    y,
                    pad_token=pad_token
                )

                correct_tokens += correct
                total_tokens += total

                if pad_token is not None:
                    mask = y != pad_token
                    sequence_correct = (
                        ((predictions == y) | ~mask)
                        .all(dim=1)
                    )
                else:
                    sequence_correct = (
                        predictions == y
                    ).all(dim=1)

                exact_matches += sequence_correct.sum().item()
                total_sequences += y.size(0)

        token_accuracy = 100 * correct_tokens / total_tokens
        exact_match_accuracy = 100 * exact_matches / total_sequences

        results.append(
            (
                task,
                train_length,
                length,
                token_accuracy,
                exact_match_accuracy
            )
        )

        print(f"Token accuracy: {token_accuracy:.2f}%")
        print(f"Exact match accuracy: {exact_match_accuracy:.2f}%")

    os.makedirs("outputs/results", exist_ok=True)

    results_path = (
        f"outputs/results/"
        f"{task}_train{train_length}_baseline_results.csv"
    )

    with open(results_path, "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "Task",
            "Train Length",
            "Test Length",
            "Token Accuracy",
            "Exact Match Accuracy"
        ])

        writer.writerows(results)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()