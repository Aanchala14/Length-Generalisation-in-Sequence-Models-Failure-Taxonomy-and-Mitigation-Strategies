import argparse
from pathlib import Path
from src.tasks.reverse import ReverseTask
from src.tasks.associative_recall import AssociativeRecallTask

from src.tasks.addition import AdditionTask
from src.tasks.copy import CopyTask
from src.utils.config import load_config
from src.utils.seed import set_seed


def create_task(config, length):
    task = config["task"]

    if task == "copy":
        return CopyTask(
            vocab_size=config["vocab_size"],
            sequence_length=length,
            separator_token=config.get("separator_token"),
            pad_token=config.get("pad_token")
            )

    if task == "addition":
        return AdditionTask(
            sequence_length=length,
            plus_token=config.get("plus_token", 10),
            pad_token=config.get("pad_token", 11)
            )

    if task == "reverse":
        return ReverseTask(
            vocab_size=config["vocab_size"],
            sequence_length=length
            )

    if task == "associative_recall":
        return AssociativeRecallTask(
            vocab_size=config["vocab_size"],
            sequence_length=length,
            query_token=config.get("query_token"),
            pad_token=config.get("pad_token")
           )

    raise ValueError(f"Unknown task: {task}")


def generate_split(task, samples, output_file):
    dataset = task.generate_dataset(samples)
    task.save_jsonl(dataset, output_file)

def pad_sample(sample, max_sequence_length, pad_token):
    input_padding = max_sequence_length - len(sample["input"])
    target_padding = max_sequence_length - len(sample["target"])

    if input_padding < 0 or target_padding < 0:
        raise ValueError(
            "Sample is longer than max_sequence_length"
        )

    sample["input"] = (
        sample["input"]
        + [pad_token] * input_padding
    )

    sample["target"] = (
        sample["target"]
        + [pad_token] * target_padding
    )

    return sample

def generate_mixed_train_split(config, output_dir):
    import random

    train_lengths = config["train_lengths"]
    train_samples = config["train_samples"]

    samples_per_length = train_samples // len(train_lengths)
    remainder = train_samples % len(train_lengths)

    task_name = config["task"]

    if task_name == "addition":
        max_sequence_length = (2 * max(train_lengths)) + 1
    elif task_name == "copy":
        max_sequence_length = (2 * max(train_lengths)) + 1
    else:
        max_sequence_length = max(train_lengths)

    pad_token = config.get("pad_token")

    if pad_token is None:
        raise ValueError(
            "mixed_train_file requires pad_token"
        )

    mixed_dataset = []

    for index, length in enumerate(train_lengths):
        task = create_task(
            config,
            length
        )

        n_samples = samples_per_length

        if index < remainder:
            n_samples += 1

        dataset = task.generate_dataset(n_samples)

        for sample in dataset:
            mixed_dataset.append(
                pad_sample(
                    sample,
                    max_sequence_length,
                    pad_token
                )
            )

    random.shuffle(mixed_dataset)

    task = create_task(
        config,
        max(train_lengths)
    )

    task.save_jsonl(
        mixed_dataset,
        output_dir / "train.jsonl"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/baseline.yaml"
    )
    args = parser.parse_args()

    config = load_config(args.config)

    set_seed(config.get("seed", 42))

    task_name = config["task"]

    output_dir = Path(
    config.get(
        "data_dir",
        f"data/synthetic/{task_name}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train_lengths = config.get("train_lengths")

    if train_lengths and config.get("mixed_train_file", False):
        generate_mixed_train_split(
            config,
            output_dir
        )
    elif train_lengths:
        samples_per_length = config["train_samples"] // len(train_lengths)
        remainder = config["train_samples"] % len(train_lengths)

        for index, length in enumerate(train_lengths):
            train_task = create_task(
                config,
                length
            )

            n_samples = samples_per_length

            if index < remainder:
                n_samples += 1

            generate_split(
                train_task,
                n_samples,
                output_dir / f"train_{length}.jsonl"
            )

    else:
        train_task = create_task(
            config,
            config["train_length"]
            )

        generate_split(
            train_task,
            config["train_samples"],
            output_dir / "train.jsonl"
        )
    
    validation_task = create_task(
        config,
        config["train_length"]
    )

    generate_split(
        validation_task,
        config["validation_samples"],
        output_dir / "validation.jsonl"
    )

    for length in config["test_lengths"]:
        test_task = create_task(config, length)

        generate_split(
            test_task,
            config["test_samples"],
            output_dir / f"test_{length}.jsonl"
        )

    print(f"Finished generating {task_name} datasets.")


if __name__ == "__main__":
    main()