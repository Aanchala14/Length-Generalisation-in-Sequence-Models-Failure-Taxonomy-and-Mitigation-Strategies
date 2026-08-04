from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("outputs/results/pe_pilot_results")
OUTPUT_DIR = Path("outputs/plots/pe_pilot")

TASKS = {
    "copy": {
        "label": "Delayed Copy",
        "train_length": 128,
        "files": {
            "learned": "copy_train128_learned_results.csv",
            "sinusoidal": "copy_train128_sinusoidal_results.csv",
            "none": "copy_train128_none_results.csv",
        },
    },
    "reverse": {
        "label": "Reverse",
        "train_length": 128,
        "files": {
            "learned": "reverse_train128_learned_results.csv",
            "sinusoidal": "reverse_train128_sinusoidal_results.csv",
            "none": "reverse_train128_none_results.csv",
        },
    },
    "addition": {
        "label": "Addition",
        "train_length": 16,
        "files": {
            "learned": "addition_train16_learned_results.csv",
            "sinusoidal": "addition_train16_sinusoidal_results.csv",
            "none": "addition_train16_none_results.csv",
        },
    },
}

ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "none": "NoPE",
}

COLORS = {
    "learned": "#2E86AB",
    "sinusoidal": "#3CA370",
    "none": "#D1495B",
}


def load_task_results(task_config):
    frames = []

    for encoding, file_name in task_config["files"].items():
        path = RESULT_DIR / file_name

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {path}"
            )

        frame = pd.read_csv(path)
        frame["Positional Encoding"] = encoding
        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True
    )


def plot_task_metric(task_name, task_config, metric):
    results = load_task_results(task_config)

    plt.figure(figsize=(8, 5))

    for encoding, encoding_results in results.groupby(
        "Positional Encoding"
    ):
        encoding_results = encoding_results.sort_values(
            "Test Length"
        )

        plt.plot(
            encoding_results["Test Length"],
            encoding_results[metric],
            marker="o",
            linewidth=2.5,
            label=ENCODING_LABELS[encoding],
            color=COLORS[encoding]
        )

    train_length = task_config["train_length"]

    plt.axvline(
        x=train_length,
        linestyle="--",
        color="gray",
        alpha=0.6,
        linewidth=1.5,
        label="Train length"
    )

    plt.xscale("log", base=2)

    test_lengths = sorted(results["Test Length"].unique())

    plt.xticks(
        test_lengths,
        test_lengths
    )

    plt.ylim(-2, 102)

    plt.xlabel("Test sequence length")
    plt.ylabel(metric)
    plt.title(
        f"{task_config['label']}: {metric} by Positional Encoding"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.35
    )

    plt.legend()
    plt.tight_layout()

    output_name = (
        f"{task_name}_"
        f"{metric.lower().replace(' ', '_')}_pe_comparison.png"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_DIR / output_name,
        dpi=300
    )

    plt.close()


def create_summary_table():
    rows = []

    for task_name, task_config in TASKS.items():
        results = load_task_results(task_config)

        train_length = task_config["train_length"]

        for encoding, encoding_results in results.groupby(
            "Positional Encoding"
        ):
            train_row = encoding_results[
                encoding_results["Test Length"] == train_length
            ].iloc[0]

            longest_length = encoding_results["Test Length"].max()

            long_row = encoding_results[
                encoding_results["Test Length"] == longest_length
            ].iloc[0]

            rows.append({
                "Task": task_config["label"],
                "Train Length": train_length,
                "Longest Test Length": longest_length,
                "Positional Encoding": ENCODING_LABELS[encoding],
                "Exact@Train": train_row["Exact Match Accuracy"],
                "Exact@Longest": long_row["Exact Match Accuracy"],
                "Token@Train": train_row["Token Accuracy"],
                "Token@Longest": long_row["Token Accuracy"],
            })

    summary = pd.DataFrame(rows)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        OUTPUT_DIR / "pe_pilot_summary.csv",
        index=False
    )


def main():
    for task_name, task_config in TASKS.items():
        plot_task_metric(
            task_name,
            task_config,
            "Exact Match Accuracy"
        )

        plot_task_metric(
            task_name,
            task_config,
            "Token Accuracy"
        )

    create_summary_table()

    print("Saved PE pilot plots to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()