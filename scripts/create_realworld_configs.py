import os
import yaml


DATASET = "SelfRegulationSCP2"
TRAIN_LENGTH = 256
TEST_LENGTHS = [256, 512, 1024, 1152]
POSITIONAL_ENCODINGS = ["learned", "sinusoidal", "none", "alibi", "rope"]
SEEDS = [42, 123, 2024]

CONFIG_DIR = "configs/realworld"


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)

    for positional_encoding in POSITIONAL_ENCODINGS:
        for seed in SEEDS:
            config = {
                "task": "realworld_timeseries",
                "dataset": DATASET,
                "train_length": TRAIN_LENGTH,
                "test_lengths": TEST_LENGTHS,
                "positional_encoding": positional_encoding,
                "experiment_name": f"{positional_encoding}_seed{seed}",
                "embedding_dim": 128,
                "num_heads": 4,
                "num_layers": 2,
                "feedforward_dim": 256,
                "dropout": 0.1,
                "max_length": 1152,
                "batch_size": 16,
                "epochs": 50,
                "learning_rate": 0.001,
                "seed": seed,
                "data_root": "data/realworld",
                "checkpoint_dir": "outputs/checkpoints/realworld",
                "results_dir": "outputs/results/realworld",
            }

            path = (
                f"{CONFIG_DIR}/"
                f"selfregulationscp2_{positional_encoding}_seed{seed}.yaml"
            )

            with open(path, "w") as file:
                yaml.safe_dump(config, file, sort_keys=False)

            print(f"Created {path}")


if __name__ == "__main__":
    main()