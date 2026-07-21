import argparse
from pathlib import Path
from src.tasks.reverse import ReverseTask

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

    raise ValueError(f"Unknown task: {task}")


def generate_split(task, samples, output_file):
    dataset = task.generate_dataset(samples)
    task.save_jsonl(dataset, output_file)


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

    output_dir = Path(f"data/synthetic/{task_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

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