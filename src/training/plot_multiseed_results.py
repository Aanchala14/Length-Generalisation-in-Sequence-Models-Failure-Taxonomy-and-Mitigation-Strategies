from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_DIR = Path("outputs/results/multiseed_results")
OUTPUT_DIR = Path("outputs/plots/multiseed")

TASK_LABELS = {
    "copy": "Delayed Copy",
    "reverse": "Reverse",
    "addition": "Addition",
}

TASK_ORDER = [
    "copy",
    "reverse",
    "addition",
]

ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "none": "NoPE",
    "alibi": "ALiBi",
    "rope": "RoPE",
}

ENCODING_ORDER = [
    "learned",
    "sinusoidal",
    "none",
    "alibi",
    "rope",
]

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


def load_results():
    files = sorted(RESULT_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No result CSV files found in {RESULT_DIR}"
        )

    frames = [
        pd.read_csv(file)
        for file in files
    ]

    results = pd.concat(
        frames,
        ignore_index=True
    )

    required_columns = {
        "Task",
        "Train Length",
        "Test Length",
        "Positional Encoding",
        "Seed",
        "Token Accuracy",
        "Exact Match Accuracy",
    }

    missing_columns = required_columns - set(results.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in result files: {missing_columns}"
        )

    return results


def add_error_rate_columns(results):
    results = results.copy()

    results["Token Error Rate"] = (
        100.0 - results["Token Accuracy"]
    )

    results["Exact Match Error Rate"] = (
        100.0 - results["Exact Match Accuracy"]
    )

    return results


def aggregate_by_length(results):
    grouped = results.groupby(
        [
            "Task",
            "Train Length",
            "Test Length",
            "Positional Encoding",
        ],
        as_index=False
    )

    aggregate = grouped.agg(
        Seed_Count=("Seed", "nunique"),
        Token_Mean=("Token Accuracy", "mean"),
        Token_Std=("Token Accuracy", "std"),
        Token_SE=("Token Accuracy", "sem"),
        Token_Error_Mean=("Token Error Rate", "mean"),
        Token_Error_Std=("Token Error Rate", "std"),
        Token_Error_SE=("Token Error Rate", "sem"),
        Exact_Mean=("Exact Match Accuracy", "mean"),
        Exact_Std=("Exact Match Accuracy", "std"),
        Exact_SE=("Exact Match Accuracy", "sem"),
        Exact_Error_Mean=("Exact Match Error Rate", "mean"),
        Exact_Error_Std=("Exact Match Error Rate", "std"),
        Exact_Error_SE=("Exact Match Error Rate", "sem"),
    )

    return aggregate.fillna(0.0)


def get_train_and_long_rows(aggregate):
    rows = []

    for task in TASK_ORDER:
        task_results = aggregate[aggregate["Task"] == task]

        for encoding in ENCODING_ORDER:
            encoding_results = task_results[
                task_results["Positional Encoding"] == encoding
            ].sort_values("Test Length")

            if encoding_results.empty:
                continue

            train_length = int(
                encoding_results["Train Length"].iloc[0]
            )

            train_row = encoding_results[
                encoding_results["Test Length"] == train_length
            ].iloc[0]

            longest_row = encoding_results[
                encoding_results["Test Length"]
                == encoding_results["Test Length"].max()
            ].iloc[0]

            rows.append({
                "Task": task,
                "Task Label": TASK_LABELS[task],
                "Train Length": train_length,
                "Longest Test Length": int(longest_row["Test Length"]),
                "Positional Encoding": encoding,
                "Positional Encoding Label": ENCODING_LABELS[encoding],
                "Seeds": int(train_row["Seed_Count"]),
                "Exact@Train Mean": train_row["Exact_Mean"],
                "Exact@Train Std": train_row["Exact_Std"],
                "Exact@Train SE": train_row["Exact_SE"],
                "Exact@Train Error Mean": train_row["Exact_Error_Mean"],
                "Exact@Train Error Std": train_row["Exact_Error_Std"],
                "Exact@Train Error SE": train_row["Exact_Error_SE"],
                "Exact@Longest Mean": longest_row["Exact_Mean"],
                "Exact@Longest Std": longest_row["Exact_Std"],
                "Exact@Longest SE": longest_row["Exact_SE"],
                "Exact@Longest Error Mean": longest_row[
                    "Exact_Error_Mean"
                ],
                "Exact@Longest Error Std": longest_row[
                    "Exact_Error_Std"
                ],
                "Exact@Longest Error SE": longest_row[
                    "Exact_Error_SE"
                ],
                "Token@Train Mean": train_row["Token_Mean"],
                "Token@Train Std": train_row["Token_Std"],
                "Token@Train SE": train_row["Token_SE"],
                "Token@Longest Mean": longest_row["Token_Mean"],
                "Token@Longest Std": longest_row["Token_Std"],
                "Token@Longest SE": longest_row["Token_SE"],
                "Generalisation Gap": (
                    train_row["Exact_Mean"]
                    - longest_row["Exact_Mean"]
                ),
                "Failure Length <10%": get_failure_length(
                    encoding_results,
                    train_length,
                    threshold=10.0
                ),
                "Failure Type": classify_failure(
                    train_row["Exact_Mean"],
                    longest_row["Exact_Mean"]
                ),
            })

    return pd.DataFrame(rows)


def get_failure_length(encoding_results, train_length, threshold):
    extrapolation_rows = encoding_results[
        encoding_results["Test Length"] > train_length
    ].sort_values("Test Length")

    failures = extrapolation_rows[
        extrapolation_rows["Exact_Mean"] < threshold
    ]

    if failures.empty:
        return "Not below threshold"

    return int(failures["Test Length"].iloc[0])


def classify_failure(train_exact, long_exact):
    if train_exact < 10.0:
        return "Underfits training length"

    if train_exact >= 90.0 and long_exact < 10.0:
        return "Fits training length; fails extrapolation"

    if long_exact < 10.0:
        return "Partial fit; fails extrapolation"

    return "Maintains extrapolation"


def save_line_plot(aggregate, task, metric_prefix):
    task_results = aggregate[aggregate["Task"] == task]

    fig, ax = plt.subplots(figsize=(9, 5.4))

    for encoding in ENCODING_ORDER:
        encoding_results = task_results[
            task_results["Positional Encoding"] == encoding
        ].sort_values("Test Length")

        if encoding_results.empty:
            continue

        mean_col = f"{metric_prefix}_Mean"
        se_col = f"{metric_prefix}_SE"

        ax.errorbar(
            encoding_results["Test Length"],
            encoding_results[mean_col],
            yerr=encoding_results[se_col],
            label=ENCODING_LABELS[encoding],
            color=COLORS[encoding],
            marker=MARKERS[encoding],
            linestyle=LINE_STYLES[encoding],
            linewidth=2.2,
            markersize=6,
            capsize=3,
        )

    train_length = int(task_results["Train Length"].iloc[0])

    ax.axvline(
        train_length,
        color="gray",
        linestyle="--",
        linewidth=1.3,
        alpha=0.7,
        label="Train length",
    )

    ax.set_xscale("log", base=2)
    lengths = sorted(task_results["Test Length"].unique())
    ax.set_xticks(lengths)
    ax.set_xticklabels(lengths)
    ax.set_ylim(-3, 103)
    ax.set_xlabel("Test sequence length")
    ax.set_ylabel(f"{metric_prefix.replace('_', ' ')} (%)")
    ax.set_title(
        f"{TASK_LABELS[task]}: {metric_prefix.replace('_', ' ')} degradation"
    )
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    fig.tight_layout()

    output_name = (
        f"{task}_{metric_prefix.lower()}_degradation_mean_se.png"
    )

    fig.savefig(
        OUTPUT_DIR / output_name,
        dpi=300
    )

    plt.close(fig)


def save_summary_bar(summary, column, error_column, output_name, title, ylabel):
    fig, axes = plt.subplots(
        1,
        len(TASK_ORDER),
        figsize=(15.5, 4.8),
        sharey=True
    )

    for ax, task in zip(axes, TASK_ORDER):
        task_summary = summary[
            summary["Task"] == task
        ].set_index("Positional Encoding").loc[
            ENCODING_ORDER
        ].reset_index()

        x_positions = range(len(task_summary))

        ax.bar(
            x_positions,
            task_summary[column],
            yerr=task_summary[error_column],
            color=[
                COLORS[encoding]
                for encoding in task_summary["Positional Encoding"]
            ],
            width=0.72,
            capsize=4,
        )

        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(
            task_summary["Positional Encoding Label"],
            rotation=35,
            ha="right"
        )
        ax.set_ylim(0, 105)
        ax.grid(
            True,
            axis="y",
            linestyle="--",
            alpha=0.35
        )

        for x, value in zip(x_positions, task_summary[column]):
            ax.text(
                x,
                min(value + 2.5, 101.0),
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8
            )

    axes[0].set_ylabel(ylabel)
    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / output_name,
        dpi=300
    )

    plt.close(fig)


def save_gap_plot(summary):
    fig, axes = plt.subplots(
        1,
        len(TASK_ORDER),
        figsize=(15.5, 4.8),
        sharey=True
    )

    for ax, task in zip(axes, TASK_ORDER):
        task_summary = summary[
            summary["Task"] == task
        ].set_index("Positional Encoding").loc[
            ENCODING_ORDER
        ].reset_index()

        x_positions = range(len(task_summary))

        ax.bar(
            x_positions,
            task_summary["Generalisation Gap"],
            color=[
                COLORS[encoding]
                for encoding in task_summary["Positional Encoding"]
            ],
            width=0.72,
        )

        ax.set_title(TASK_LABELS[task])
        ax.set_xticks(list(x_positions))
        ax.set_xticklabels(
            task_summary["Positional Encoding Label"],
            rotation=35,
            ha="right"
        )
        ax.set_ylim(0, 105)
        ax.grid(
            True,
            axis="y",
            linestyle="--",
            alpha=0.35
        )

    axes[0].set_ylabel("Exact-match gap (train - longest)")
    fig.suptitle("Generalisation Gap by Task and Positional Encoding")
    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "generalisation_gap_mean.png",
        dpi=300
    )

    plt.close(fig)


def save_baseline_comparison(summary):
    rows = []

    for task in TASK_ORDER:
        task_summary = summary[summary["Task"] == task]

        baseline = task_summary[
            task_summary["Positional Encoding"] == "learned"
        ].iloc[0]

        for _, row in task_summary.iterrows():
            rows.append({
                "Task": row["Task Label"],
                "Positional Encoding": row[
                    "Positional Encoding Label"
                ],
                "Delta Exact@Train vs Baseline": (
                    row["Exact@Train Mean"]
                    - baseline["Exact@Train Mean"]
                ),
                "Delta Exact@Longest vs Baseline": (
                    row["Exact@Longest Mean"]
                    - baseline["Exact@Longest Mean"]
                ),
                "Delta Gap vs Baseline": (
                    row["Generalisation Gap"]
                    - baseline["Generalisation Gap"]
                ),
            })

    comparison = pd.DataFrame(rows)

    comparison.to_csv(
        OUTPUT_DIR / "baseline_comparison.csv",
        index=False
    )

    return comparison


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = load_results()
    results = add_error_rate_columns(results)

    aggregate = aggregate_by_length(results)
    summary = get_train_and_long_rows(aggregate)

    aggregate.to_csv(
        OUTPUT_DIR / "multiseed_aggregate_by_length.csv",
        index=False
    )

    summary.to_csv(
        OUTPUT_DIR / "multiseed_summary.csv",
        index=False
    )

    summary[
        [
            "Task Label",
            "Train Length",
            "Longest Test Length",
            "Positional Encoding Label",
            "Exact@Train Mean",
            "Exact@Train Std",
            "Exact@Train SE",
            "Exact@Longest Mean",
            "Exact@Longest Std",
            "Exact@Longest SE",
            "Generalisation Gap",
            "Failure Length <10%",
            "Failure Type",
        ]
    ].to_csv(
        OUTPUT_DIR / "failure_length_table.csv",
        index=False
    )

    save_baseline_comparison(summary)

    for task in TASK_ORDER:
        save_line_plot(
            aggregate,
            task,
            "Exact"
        )

        save_line_plot(
            aggregate,
            task,
            "Token"
        )

    save_summary_bar(
        summary,
        "Exact@Train Mean",
        "Exact@Train SE",
        "exact_at_train_mean_se.png",
        "Exact Match Accuracy at Training Length",
        "Exact match accuracy (%)"
    )

    save_summary_bar(
        summary,
        "Exact@Longest Mean",
        "Exact@Longest SE",
        "exact_at_longest_mean_se.png",
        "Exact Match Accuracy at Longest Test Length",
        "Exact match accuracy (%)"
    )

    save_summary_bar(
        summary,
        "Token@Train Mean",
        "Token@Train SE",
        "token_at_train_mean_se.png",
        "Token Accuracy at Training Length",
        "Token accuracy (%)"
    )

    save_summary_bar(
        summary,
        "Token@Longest Mean",
        "Token@Longest SE",
        "token_at_longest_mean_se.png",
        "Token Accuracy at Longest Test Length",
        "Token accuracy (%)"
    )

    save_gap_plot(summary)

    print("Saved multi-seed plots and tables to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
