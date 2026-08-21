import argparse
import csv
import math
import os

import torch

from src.data.dataloader import get_dataloader
from src.models.transformer import TransformerModel
from src.utils.config import load_config


def get_sequence_length(task, length):
    if task == "addition":
        return (2 * length) + 1

    return length


def attention_entropy(attention):
    eps = 1e-9

    entropy = -(
        attention * torch.log(attention + eps)
    ).sum(dim=-1)

    return entropy.mean().item()


def normalised_attention_entropy(attention):
    sequence_length = attention.size(-1)

    entropy = attention_entropy(attention)

    return entropy / math.log(sequence_length)


def average_attention_distance(attention):
    sequence_length = attention.size(-1)
    device = attention.device

    positions = torch.arange(
        sequence_length,
        device=device
    )

    distances = (
        positions[None, :]
        - positions[:, None]
    ).abs().float()

    while distances.dim() < attention.dim():
        distances = distances.unsqueeze(0)

    avg_distance = (
        attention * distances
    ).sum(dim=-1).mean()

    return avg_distance.item()


def local_attention_ratio(attention, window_size=10):
    sequence_length = attention.size(-1)
    device = attention.device

    positions = torch.arange(
        sequence_length,
        device=device
    )

    distances = (
        positions[None, :]
        - positions[:, None]
    ).abs()

    local_mask = distances <= window_size

    while local_mask.dim() < attention.dim():
        local_mask = local_mask.unsqueeze(0)

    ratio = attention.masked_fill(
        ~local_mask,
        0.0
    ).sum(dim=-1).mean()

    return ratio.item()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True
    )

    parser.add_argument(
        "--length",
        type=int,
        required=True
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=5
    )

    args = parser.parse_args()
    config = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    task = config["task"]
    train_length = config["train_length"]

    positional_encoding = config.get(
        "positional_encoding",
        "learned"
    )

    seed = config.get("seed", 42)

    experiment_name = config.get(
        "experiment_name",
        f"{positional_encoding}_seed{seed}"
    )

    checkpoint_dir = config.get(
        "checkpoint_dir",
        "outputs/checkpoints"
    )

    data_dir = config.get(
        "data_dir",
        f"data/synthetic/{task}"
    )

    max_test_sequence_length = max(
        get_sequence_length(task, length)
        for length in config["test_lengths"]
    )

    max_train_sequence_length = get_sequence_length(
        task,
        train_length
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
        max_length=max_length,
        positional_encoding=positional_encoding
    )

    checkpoint_path = (
        f"{checkpoint_dir}/"
        f"{task}_train{train_length}_{experiment_name}.pt"
    )

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    model.to(device)
    model.eval()

    loader = get_dataloader(
        f"{data_dir}/test_{args.length}.jsonl",
        batch_size=config["batch_size"],
        shuffle=False
    )

    rows = []

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(loader):
            if batch_index >= args.max_batches:
                break

            x = x.to(device)

            logits, attention_maps = model(
                x,
                return_attention=True
            )

            for layer_index, attention in enumerate(attention_maps):
                rows.append({
                    "Task": task,
                    "Train Length": train_length,
                    "Test Length": args.length,
                    "Positional Encoding": positional_encoding,
                    "Seed": seed,
                    "Layer": layer_index,
                    "Attention Entropy": attention_entropy(attention),
                    "Normalised Attention Entropy": normalised_attention_entropy(attention),
                    "Average Attention Distance": average_attention_distance(attention),
                    "Local Attention Ratio": local_attention_ratio(attention),
                })

    output_dir = "outputs/analysis/attention"
    os.makedirs(output_dir, exist_ok=True)

    output_path = (
        f"{output_dir}/"
        f"{task}_train{train_length}_{positional_encoding}_"
        f"seed{seed}_length{args.length}_attention.csv"
    )

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved attention analysis to: {output_path}")


if __name__ == "__main__":
    main()