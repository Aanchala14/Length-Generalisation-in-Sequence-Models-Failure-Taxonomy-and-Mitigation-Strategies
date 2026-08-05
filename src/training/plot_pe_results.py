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
            "alibi": "copy_train128_alibi_results.csv",
            "rope": "copy_train128_rope_results.csv",
        },
    },
    "reverse": {
        "label": "Reverse",
        "train_length": 128,
        "files": {
            "learned": "reverse_train128_learned_results.csv",
            "sinusoidal": "reverse_train128_sinusoidal_results.csv",
            "none": "reverse_train128_none_results.csv",
            "alibi": "reverse_train128_alibi_results.csv",
            "rope": "reverse_train128_rope_results.csv",
        },
    },
    "addition": {
        "label": "Addition",
        "train_length": 16,
        "files": {
            "learned": "addition_train16_learned_results.csv",
            "sinusoidal": "addition_train16_sinusoidal_results.csv",
            "none": "addition_train16_none_results.csv",
            "alibi": "addition_train16_alibi_results.csv",
            "rope": "addition_train16_rope_results.csv",
        },
    },
}

ENCODING_ORDER = [
    "learned",
    "sinusoidal",
    "none",
    "alibi",
    "rope",
]

ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "none": "NoPE",
    "alibi": "ALiBi",
    "rope": "RoPE",
}

COLORS = {
    "learned": "#2E86AB",
    "sinusoidal": "#3CA370",
    "none": "#D1495B",
    "alibi": "#F28E2B",
    "rope": "#6F4EAD",
}

MARKERS = {
    "learned": "o",
    "sinusoidal": "s",
    "none": "D",
    "alibi": "^",
    "rope": "P",
}

LINE_STYLES = {
    "learned": "-",
    "sinusoidal": "--",
    "none": "-.",
    "alibi": ":",
    "rope": (0, (3, 1, 1, 1)),
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

    for encoding in ENCODING_ORDER:
        encoding_results = results[
            results["Positional Encoding"] == encoding
        ]

        if encoding_results.empty:
            continue

        encoding_results = encoding_results.sort_values(
            "Test Length"
        )

        plt.plot(
            encoding_results["Test Length"],
            encoding_results[metric],
            marker=MARKERS[encoding],
            linestyle=LINE_STYLES[encoding],
            linewidth=2.5,
            markersize=7,
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

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0
    )

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

        for encoding in ENCODING_ORDER:
            encoding_results = results[
                results["Positional Encoding"] == encoding
            ]

            if encoding_results.empty:
                continue

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

    return summary


def plot_summary_bar(summary, column, output_name, title, ylabel):
    fig, axes = plt.subplots(
        1,
        len(TASKS),
        figsize=(15, 4.8),
        sharey=True
    )

    for axis, task_config in zip(axes, TASKS.values()):
        task_summary = summary[
            summary["Task"] == task_config["label"]
        ].copy()

        task_summary["encoding_key"] = task_summary[
            "Positional Encoding"
        ].map({
            label: key
            for key, label in ENCODING_LABELS.items()
        })

        task_summary = task_summary.set_index(
            "encoding_key"
        ).loc[ENCODING_ORDER].reset_index()

        x_positions = range(len(task_summary))

        bars = axis.bar(
            x_positions,
            task_summary[column],
            color=[
                COLORS[encoding]
                for encoding in task_summary["encoding_key"]
            ],
            width=0.72
        )

        axis.set_title(task_config["label"])
        axis.set_xticks(list(x_positions))
        axis.set_xticklabels(
            task_summary["Positional Encoding"],
            rotation=35,
            ha="right"
        )

        axis.grid(
            True,
            axis="y",
            linestyle="--",
            alpha=0.35
        )

        for bar in bars:
            height = bar.get_height()

            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + 1.2,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    axes[0].set_ylabel(ylabel)

    for axis in axes:
        axis.set_ylim(0, 105)

    fig.suptitle(title)
    fig.tight_layout()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_DIR / output_name,
        dpi=300
    )

    plt.close(fig)


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

    summary = create_summary_table()

    plot_summary_bar(
        summary,
        "Exact@Train",
        "pe_exact_at_train_length.png",
        "Exact Match Accuracy at Training Length",
        "Exact match accuracy (%)"
    )

    plot_summary_bar(
        summary,
        "Exact@Longest",
        "pe_exact_at_1024.png",
        "Exact Match Accuracy at Longest Test Length",
        "Exact match accuracy (%)"
    )

    plot_summary_bar(
        summary,
        "Token@Train",
        "pe_token_at_train_length.png",
        "Token Accuracy at Training Length",
        "Token accuracy (%)"
    )

    plot_summary_bar(
        summary,
        "Token@Longest",
        "pe_token_at_1024.png",
        "Token Accuracy at Longest Test Length",
        "Token accuracy (%)"
    )

    print("Saved PE pilot plots to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
