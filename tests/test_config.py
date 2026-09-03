from pathlib import Path

from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "multiseed"
    / "addition_learned_seed42.yaml"
)


def test_load_config():
    config = load_config(CONFIG_PATH)

    assert isinstance(config, dict)
    assert config["task"] == "addition"
    assert config["train_length"] == 16
    assert config["test_lengths"] == [
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
    ]
    assert config["positional_encoding"] == "learned"
    assert config["seed"] == 42


def test_required_configuration_fields_are_present():
    config = load_config(CONFIG_PATH)

    required_fields = {
        "task",
        "train_length",
        "test_lengths",
        "vocab_size",
        "pad_token",
        "embedding_dim",
        "num_heads",
        "num_layers",
        "feedforward_dim",
        "dropout",
        "batch_size",
        "epochs",
        "learning_rate",
        "seed",
        "positional_encoding",
        "checkpoint_dir",
        "results_dir",
        "data_dir",
    }

    assert required_fields.issubset(config.keys())