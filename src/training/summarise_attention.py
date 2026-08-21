from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ATTENTION_DIR = Path("outputs/analysis/attention")
RESULTS_DIR = Path("outputs/results/multiseed_results")
OUTPUT_DIR = Path("outputs/analysis/attention_summary")

TASK_LABELS = {
    "addition": "Addition",
    "copy": "Delayed Copy",
    "reverse": "Reverse",
}

ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "rope": "RoPE",
    "alibi": "ALiBi",
    "none": "NoPE",
}

COLORS = {
    "learned": "#2E86AB",
    "sinusoidal": "#3CA370",
    "rope": "#6F4EAD",
    "alibi": "#F28E2B",
    "none": "#D1495B",
}


def standard_error(series):
    if len(series) <= 1:
        return 0.0

    return series.std() / (len(series) ** 0.5)


def load_attention_files():
    files = sorted(ATTENTION_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No attention CSV files found in {ATTENTION_DIR}"
        )

    frames = []

    for file in files:
        frame = pd.read_csv(file)
        frame["Source File"] = file.name
        frames.append(frame)

    return pd.concat(
        frames,
        ignore_index=True
    )


def load_accuracy_results():
    files = sorted(RESULTS_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No accuracy CSV files found in {RESULTS_DIR}"
        )

    frames = [
        pd.read_csv(file)
        for file in files
    ]

    results = pd.concat(
        frames,
        ignore_index=True
    )

    return results


def summarise_attention_by_layer(attention):
    grouped = attention.groupby(
        [
            "Task",
            "Train Length",
            "Test Length",
            "Positional Encoding",
            "Seed",
            "Layer",
        ],
        as_index=False
    )

    summary = grouped.agg(
        Attention_Entropy_Mean=("Attention Entropy", "mean"),
        Attention_Entropy_Std=("Attention Entropy", "std"),
        Attention_Entropy_SE=("Attention Entropy", standard_error),
        Normalised_Entropy_Mean=("Normalised Attention Entropy", "mean"),
        Normalised_Entropy_Std=("Normalised Attention Entropy", "std"),
        Normalised_Entropy_SE=("Normalised Attention Entropy", standard_error),
        Average_Distance_Mean=("Average Attention Distance", "mean"),
        Average_Distance_Std=("Average Attention Distance", "std"),
        Average_Distance_SE=("Average Attention Distance", standard_error),
        Local_Ratio_Mean=("Local Attention Ratio", "mean"),
        Local_Ratio_Std=("Local Attention Ratio", "std"),
        Local_Ratio_SE=("Local Attention Ratio", standard_error),
    )

    return summary.fillna(0.0)


def summarise_attention_overall(attention):
    grouped = attention.groupby(
        [
            "Task",
            "Train Length",
            "Test Length",
            "Positional Encoding",
            "Seed",
        ],
        as_index=False
    )

    summary = grouped.agg(
        Attention_Entropy_Mean=("Attention Entropy", "mean"),
        Attention_Entropy_Std=("Attention Entropy", "std"),
        Attention_Entropy_SE=("Attention Entropy", standard_error),
        Normalised_Entropy_Mean=("Normalised Attention Entropy", "mean"),
        Normalised_Entropy_Std=("Normalised Attention Entropy", "std"),
        Normalised_Entropy_SE=("Normalised Attention Entropy", standard_error),
        Average_Distance_Mean=("Average Attention Distance", "mean"),
        Average_Distance_Std=("Average Attention Distance", "std"),
        Average_Distance_SE=("Average Attention Distance", standard_error),
        Local_Ratio_Mean=("Local Attention Ratio", "mean"),
        Local_Ratio_Std=("Local Attention Ratio", "std"),
        Local_Ratio_SE=("Local Attention Ratio", standard_error),
    )

    return summary.fillna(0.0)


def join_accuracy(attention_summary, accuracy_results):
    accuracy_subset = accuracy_results[
        [
            "Task",
            "Train Length",
            "Test Length",
            "Positional Encoding",
            "Seed",
            "Token Accuracy",
            "Exact Match Accuracy",
        ]
    ]

    joined = attention_summary.merge(
        accuracy_subset,
        on=[
            "Task",
            "Train Length",
            "Test Length",
            "Positional Encoding",
            "Seed",
        ],
        how="left"
    )

    return joined


def add_labels(frame):
    frame = frame.copy()

    frame["Task Label"] = frame["Task"].map(TASK_LABELS)
    frame["Encoding Label"] = frame["Positional Encoding"].map(
        ENCODING_LABELS
    )

    return frame


def save_metric_plot(frame, task, metric, ylabel, filename):
    task_frame = frame[
        frame["Task"] == task
    ].sort_values("Test Length")

    fig, ax = plt.subplots(figsize=(8, 5))

    for encoding in task_frame["Positional Encoding"].unique():
        encoding_frame = task_frame[
            task_frame["Positional Encoding"] == encoding
        ].sort_values("Test Length")

        label = ENCODING_LABELS.get(
            encoding,
            encoding
        )

        ax.errorbar(
            encoding_frame["Test Length"],
            encoding_frame[metric],
            yerr=encoding_frame[
                metric.replace("_Mean", "_SE")
            ],
            marker="o",
            linewidth=2,
            capsize=4,
            label=label,
            color=COLORS.get(encoding)
        )

    train_length = int(task_frame["Train Length"].iloc[0])

    ax.axvline(
        train_length,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label="Train length"
    )

    ax.set_title(
        f"{TASK_LABELS[task]}: {ylabel} vs length"
    )

    ax.set_xlabel("Test sequence length")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / filename,
        dpi=300
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    attention = load_attention_files()
    accuracy = load_accuracy_results()

    by_layer = summarise_attention_by_layer(attention)
    overall = summarise_attention_overall(attention)
    with_accuracy = join_accuracy(overall, accuracy)

    by_layer = add_labels(by_layer)
    overall = add_labels(overall)
    with_accuracy = add_labels(with_accuracy)

    by_layer.to_csv(
        OUTPUT_DIR / "attention_by_layer.csv",
        index=False
    )

    overall.to_csv(
        OUTPUT_DIR / "attention_overall.csv",
        index=False
    )

    with_accuracy.to_csv(
        OUTPUT_DIR / "attention_with_accuracy.csv",
        index=False
    )

    for task in sorted(overall["Task"].unique()):
        save_metric_plot(
            overall,
            task,
            "Normalised_Entropy_Mean",
            "Normalised attention entropy",
            f"{task}_normalised_entropy.png"
        )

        save_metric_plot(
            overall,
            task,
            "Average_Distance_Mean",
            "Average attention distance",
            f"{task}_average_attention_distance.png"
        )

        save_metric_plot(
            overall,
            task,
            "Local_Ratio_Mean",
            "Local attention ratio",
            f"{task}_local_attention_ratio.png"
        )

    print(f"Saved attention summaries to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()