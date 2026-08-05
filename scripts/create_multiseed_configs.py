from pathlib import Path


CONFIG_DIR = Path("configs/multiseed")
CONFIG_DIR.mkdir(parents=True, exist_ok=True)


TASKS = {
    "copy": {
        "train_length": 128,
        "test_lengths": [128, 256, 512, 1024],
        "train_samples": 10000,
        "validation_samples": 1000,
        "test_samples": 1000,
        "vocab_size": 102,
        "pad_token": 101,
        "separator_token": 100,
        "batch_size": 1,
        "max_length": 2049,
    },
    "reverse": {
        "train_length": 128,
        "test_lengths": [128, 256, 512, 1024],
        "train_samples": 10000,
        "validation_samples": 1000,
        "test_samples": 1000,
        "vocab_size": 100,
        "batch_size": 1,
        "max_length": 1024,
    },
    "addition": {
        "train_length": 16,
        "test_lengths": [16, 32, 64, 128, 256, 512, 1024],
        "train_samples": 10000,
        "validation_samples": 1000,
        "test_samples": 1000,
        "vocab_size": 12,
        "pad_token": 11,
        "plus_token": 10,
        "batch_size": 8,
        "max_length": 2049,
    },
}

POSITIONAL_ENCODINGS = [
    "learned",
    "sinusoidal",
    "none",
    "alibi",
    "rope",
]

SEEDS = [
    123,
    2024,
]


def yaml_list(values, indent=2):
    spaces = " " * indent
    return "\n".join(f"{spaces}- {value}" for value in values)


def create_config(task_name, task_config, positional_encoding, seed):
    experiment_name = f"{positional_encoding}_seed{seed}"

    lines = [
        f"task: {task_name}",
        "",
        f"train_length: {task_config['train_length']}",
        "",
        "test_lengths:",
        yaml_list(task_config["test_lengths"]),
        "",
        f"train_samples: {task_config['train_samples']}",
        f"validation_samples: {task_config['validation_samples']}",
        f"test_samples: {task_config['test_samples']}",
        "",
        f"vocab_size: {task_config['vocab_size']}",
    ]

    if "pad_token" in task_config:
        lines.append(f"pad_token: {task_config['pad_token']}")

    if "separator_token" in task_config:
        lines.append(f"separator_token: {task_config['separator_token']}")

    if "plus_token" in task_config:
        lines.append(f"plus_token: {task_config['plus_token']}")

    lines.extend([
        "",
        "embedding_dim: 128",
        "num_heads: 4",
        "num_layers: 2",
        "feedforward_dim: 256",
        "dropout: 0.1",
        f"max_length: {task_config['max_length']}",
        "",
        f"batch_size: {task_config['batch_size']}",
        "epochs: 10",
        "learning_rate: 0.001",
        "",
        f"seed: {seed}",
        f"positional_encoding: {positional_encoding}",
        f"experiment_name: {experiment_name}",
        "checkpoint_dir: outputs/checkpoints/multiseed_checkpoints",
        "results_dir: outputs/results/multiseed_results",
        f"data_dir: data/synthetic/multiseed/{task_name}/{experiment_name}",
        "",
    ])

    return "\n".join(lines)


def main():
    count = 0

    for task_name, task_config in TASKS.items():
        for positional_encoding in POSITIONAL_ENCODINGS:
            for seed in SEEDS:
                config_text = create_config(
                    task_name,
                    task_config,
                    positional_encoding,
                    seed
                )

                output_path = (
                    CONFIG_DIR
                    / f"{task_name}_{positional_encoding}_seed{seed}.yaml"
                )

                output_path.write_text(config_text)
                count += 1

    print(f"Created {count} multi-seed config files in {CONFIG_DIR}")


if __name__ == "__main__":
    main()