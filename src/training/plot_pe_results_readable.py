from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("outputs/results/pe_pilot_results")
OUTPUT_DIR = Path("outputs/plots/pe_pilot_readable")

TASKS = {
    "copy": {
        "label": "Delayed Copy",
        "train_length": 128,
        "files": {
            "learned": "copy_train128_learned_results.csv",
            "sinusoidal": "copy_train128_sinusoidal_results.csv",
            "none": "copy_train128_none_results.csv",
            "alibi": "copy_train128_alibi_results.csv",
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
        },
    },
}

ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "none": "NoPE",
    "alibi": "ALiBi",
}

ENCODING_ORDER = [
    "learned",
    "sinusoidal",
    "none",
    "alibi",
]

COLORS = {
    "learned": "#2E86AB",
    "sinusoidal": "#3CA370",
    "none": "#D1495B",
    "alibi": "#F28E2B",
}


def load_task_results(task_config):
    frames = []

    for encoding, file_name in task_config["files"].items():
        path = RESULT_DIR / file_name

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

        frame = pd.read_csv(path)
        frame["Positional Encoding"] = encoding
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def create_summary():
    rows = []

    for task_name, task_config in TASKS.items():
        results = load_task_results(task_config)
        train_length = task_config["train_length"]
        longest_length = results["Test Length"].max()

        for encoding in ENCODING_ORDER:
            encoding_results = results[
                results["Positional Encoding"] == encoding
            ]

            train_row = encoding_results[
                encoding_results["Test Length"] == train_length
            ].iloc[0]

            long_row = encoding_results[
                encoding_results["Test Length"] == longest_length
            ].iloc[0]

            rows.append({
                "Task": task_config["label"],
                "Encoding": ENCODING_LABELS[encoding],
                "Exact@Train": train_row["Exact Match Accuracy"],
                "Exact@Longest": long_row["Exact Match Accuracy"],
                "Token@Train": train_row["Token Accuracy"],
                "Token@Longest": long_row["Token Accuracy"],
            })

    summary = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "pe_pilot_readable_summary.csv", index=False)

    return summary


def plot_grouped_bar(summary, metric, output_name, title):
    tasks = summary["Task"].unique()
    x = range(len(tasks))
    width = 0.18

    plt.figure(figsize=(10, 5.5))

    for i, encoding in enumerate(ENCODING_LABELS.values()):
        values = []

        for task in tasks:
            row = summary[
                (summary["Task"] == task)
                & (summary["Encoding"] == encoding)
            ].iloc[0]

            values.append(row[metric])

        offsets = [
            pos + (i - 1.5) * width
            for pos in x
        ]

        plt.bar(
            offsets,
            values,
            width=width,
            label=encoding,
            color=COLORS[
                [k for k, v in ENCODING_LABELS.items() if v == encoding][0]
            ]
        )

    plt.xticks(list(x), tasks)
    plt.ylim(0, 105)
    plt.ylabel(metric)
    plt.title(title, fontweight="bold")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=300)
    plt.close()


def plot_task_lines(task_name, task_config, metric):
    results = load_task_results(task_config)

    plt.figure(figsize=(8.5, 5.3))

    markers = {
        "learned": "o",
        "sinusoidal": "s",
        "none": "^",
        "alibi": "D",
    }

    linestyles = {
        "learned": "-",
        "sinusoidal": "--",
        "none": "-.",
        "alibi": ":",
    }

    for encoding in ENCODING_ORDER:
        encoding_results = results[
            results["Positional Encoding"] == encoding
        ].sort_values("Test Length")

        plt.plot(
            encoding_results["Test Length"],
            encoding_results[metric],
            marker=markers[encoding],
            linestyle=linestyles[encoding],
            linewidth=2.5,
            markersize=7,
            label=ENCODING_LABELS[encoding],
            color=COLORS[encoding],
        )

    train_length = task_config["train_length"]

    plt.axvline(
        x=train_length,
        color="gray",
        linestyle="--",
        alpha=0.6,
        linewidth=1.4,
    )

    plt.xscale("log", base=2)

    test_lengths = sorted(results["Test Length"].unique())
    plt.xticks(test_lengths, test_lengths)

    if metric == "Exact Match Accuracy":
        plt.ylim(-3, 105)
    else:
        max_value = results[metric].max()
        plt.ylim(-2, min(105, max_value + 12))

    plt.xlabel("Test sequence length")
    plt.ylabel(metric)
    plt.title(
        f"{task_config['label']}: {metric}",
        fontweight="bold"
    )
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    file_name = (
        f"{task_name}_"
        f"{metric.lower().replace(' ', '_')}_readable.png"
    )

    plt.savefig(OUTPUT_DIR / file_name, dpi=300)
    plt.close()


def main():
    summary = create_summary()

    plot_grouped_bar(
        summary,
        metric="Exact@Train",
        output_name="pe_exact_at_train_bar.png",
        title="Does Each Positional Encoding Learn the Training Length?"
    )

    plot_grouped_bar(
        summary,
        metric="Exact@Longest",
        output_name="pe_exact_at_longest_bar.png",
        title="Does Performance Extrapolate to the Longest Test Length?"
    )

    plot_grouped_bar(
        summary,
        metric="Token@Train",
        output_name="pe_token_at_train_bar.png",
        title="Token Accuracy at Training Length"
    )

    for task_name, task_config in TASKS.items():
        plot_task_lines(
            task_name,
            task_config,
            "Exact Match Accuracy"
        )

        plot_task_lines(
            task_name,
            task_config,
            "Token Accuracy"
        )

    print(f"Saved readable PE plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()