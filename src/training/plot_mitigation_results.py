from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


INPUT_PATH = Path(
    "outputs/analysis/mitigation_summary/mitigation_summary.csv"
)

OUTPUT_DIR = Path(
    "outputs/analysis/mitigation_summary"
)

OUTPUT_PATH = OUTPUT_DIR / "mitigation_exact_train_vs_1024.png"


TASK_LABELS = {
    "copy": "Delayed Copy",
    "reverse": "Reverse",
    "addition": "Addition",
}


def shorten_experiment_name(name):
    replacements = {
        "Copy ": "",
        "Reverse ": "",
        "Addition ": "",
        "Sinusoidal": "Sin",
        "Randomised": "Rand",
        "Curriculum": "Curr",
        "Baseline": "Base",
        "Single Control": "Control",
        "Mixed Learned V2": "Mixed V2",
        "Mixed Learned": "Mixed",
    }

    short_name = name

    for old, new in replacements.items():
        short_name = short_name.replace(old, new)

    return short_name


def load_summary():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing mitigation summary file: {INPUT_PATH}"
        )

    return pd.read_csv(INPUT_PATH)


def save_plot(summary):
    summary = summary.copy()
    summary["Experiment Short"] = summary["Experiment"].apply(
        shorten_experiment_name
    )

    tasks = [
        task
        for task in ["addition", "copy", "reverse"]
        if task in summary["Task"].unique()
    ]

    fig, axes = plt.subplots(
        nrows=len(tasks),
        ncols=1,
        figsize=(12, 3.8 * len(tasks)),
        sharey=True
    )

    if len(tasks) == 1:
        axes = [axes]

    for ax, task in zip(axes, tasks):
        task_frame = summary[
            summary["Task"] == task
        ].copy()

        task_frame = task_frame.sort_values(
            ["Positional Encoding", "Experiment"]
        )

        x_positions = range(len(task_frame))

        ax.bar(
            [x - 0.18 for x in x_positions],
            task_frame["Exact@Train"],
            width=0.36,
            label="Exact@Train",
            color="#2E86AB"
        )

        ax.bar(
            [x + 0.18 for x in x_positions],
            task_frame["Exact@1024"],
            width=0.36,
            label="Exact@1024",
            color="#D1495B"
        )

        for x, (_, row) in zip(
            x_positions,
            task_frame.iterrows()
        ):
            ax.text(
                x - 0.18,
                row["Exact@Train"] + 2,
                f"{row['Exact@Train']:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

            ax.text(
                x + 0.18,
                row["Exact@1024"] + 2,
                f"{row['Exact@1024']:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

        ax.set_title(
            TASK_LABELS.get(task, task)
        )

        ax.set_ylabel("Exact match accuracy (%)")
        ax.set_ylim(0, 108)
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(
            task_frame["Experiment Short"],
            rotation=30,
            ha="right"
        )

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.35
        )

    axes[0].legend(
        loc="upper right"
    )

    fig.suptitle(
        "Mitigation Results: Training Length vs Length 1024",
        fontsize=16
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fig.savefig(
        OUTPUT_PATH,
        dpi=300
    )

    plt.close(fig)


def main():
    summary = load_summary()
    save_plot(summary)

    print(f"Saved plot to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()