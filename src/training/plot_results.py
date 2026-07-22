from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_FILES = [
    "outputs/results/copy_train128_baseline_results.csv",
    "outputs/results/reverse_train128_baseline_results.csv",
    "outputs/results/addition_train128_baseline_results.csv",
    "outputs/results/associative_recall_train128_baseline_results.csv",
]


TASK_LABELS = {
    "copy": "Delayed Copy",
    "reverse": "Reverse",
    "addition": "Addition",
    "associative_recall": "Associative Recall",
}


TASK_ORDER = [
    "copy",
    "reverse",
    "addition",
    "associative_recall",
]


COLORS = {
    "copy": "#2E86AB",
    "reverse": "#D1495B",
    "addition": "#3CA370",
    "associative_recall": "#F28E2B",
}


def load_results():
    frames = []

    for file_path in RESULT_FILES:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {file_path}"
            )

        frames.append(pd.read_csv(path))

    return pd.concat(
        frames,
        ignore_index=True
    )


def save_small_multiples(results, metric, output_path, title):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10, 7),
        sharex=True
    )

    axes = axes.flatten()

    for ax, task in zip(axes, TASK_ORDER):
        task_results = (
            results[results["Task"] == task]
            .sort_values("Test Length")
        )

        ax.plot(
            task_results["Test Length"],
            task_results[metric],
            marker="o",
            linewidth=2.5,
            color=COLORS[task]
        )

        ax.axvline(
            x=128,
            color="gray",
            linestyle="--",
            linewidth=1,
            alpha=0.6
        )

        ax.set_title(
            TASK_LABELS[task],
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024])
        ax.set_xticklabels([128, 256, 512, 1024])
        ax.grid(True, linestyle="--", alpha=0.35)

        max_value = task_results[metric].max()

        if max_value <= 15:
            ax.set_ylim(-0.5, 15)
        else:
            ax.set_ylim(-5, 105)

    fig.suptitle(
        title,
        fontsize=15,
        fontweight="bold"
    )

    fig.supxlabel("Test sequence length")
    fig.supylabel(metric)

    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


def save_generalisation_gap(results, output_path):
    rows = []

    for task in TASK_ORDER:
        task_results = results[results["Task"] == task]

        train_score = task_results[
            task_results["Test Length"] == 128
        ]["Exact Match Accuracy"].iloc[0]

        long_score = task_results[
            task_results["Test Length"] == 1024
        ]["Exact Match Accuracy"].iloc[0]

        rows.append({
            "Task": TASK_LABELS[task],
            "Train length 128": train_score,
            "Test length 1024": long_score,
            "Gap": train_score - long_score
        })

    gap_df = pd.DataFrame(rows)

    x = range(len(gap_df))

    plt.figure(figsize=(9, 5))

    plt.bar(
        [i - 0.18 for i in x],
        gap_df["Train length 128"],
        width=0.36,
        label="Length 128",
        color="#4C78A8"
    )

    plt.bar(
        [i + 0.18 for i in x],
        gap_df["Test length 1024"],
        width=0.36,
        label="Length 1024",
        color="#F58518"
    )

    plt.xticks(
        list(x),
        gap_df["Task"],
        rotation=15,
        ha="right"
    )

    plt.ylim(0, 105)
    plt.ylabel("Exact Match Accuracy")
    plt.title(
        "Generalisation Gap: Training Length vs Long Test Length",
        fontweight="bold"
    )
    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.35
    )
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.savefig(output_path, dpi=300)
    plt.close()

    gap_df.to_csv(
        "outputs/plots/baseline_generalisation_gap.csv",
        index=False
    )


def save_summary_table(results):
    rows = []

    for task in TASK_ORDER:
        task_results = results[results["Task"] == task]

        exact_128 = task_results[
            task_results["Test Length"] == 128
        ]["Exact Match Accuracy"].iloc[0]

        exact_1024 = task_results[
            task_results["Test Length"] == 1024
        ]["Exact Match Accuracy"].iloc[0]

        token_128 = task_results[
            task_results["Test Length"] == 128
        ]["Token Accuracy"].iloc[0]

        token_1024 = task_results[
            task_results["Test Length"] == 1024
        ]["Token Accuracy"].iloc[0]

        if exact_128 >= 95 and exact_1024 < 5:
            status = "Learns training length; fails extrapolation"
        elif exact_128 < 5:
            status = "Does not learn training length"
        else:
            status = "Partially learned; needs diagnosis"

        rows.append({
            "Task": TASK_LABELS[task],
            "Exact@128": exact_128,
            "Exact@1024": exact_1024,
            "Token@128": token_128,
            "Token@1024": token_1024,
            "Interpretation": status
        })

    summary = pd.DataFrame(rows)

    output_path = Path(
        "outputs/plots/baseline_task_summary.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        output_path,
        index=False
    )


def main():
    results = load_results()

    save_small_multiples(
        results,
        metric="Exact Match Accuracy",
        output_path="outputs/plots/baseline_exact_match_small_multiples.png",
        title="Exact Match Accuracy by Task"
    )

    save_small_multiples(
        results,
        metric="Token Accuracy",
        output_path="outputs/plots/baseline_token_accuracy_small_multiples.png",
        title="Token Accuracy by Task"
    )

    save_generalisation_gap(
        results,
        output_path="outputs/plots/baseline_generalisation_gap.png"
    )

    save_summary_table(results)

    print("Saved:")
    print("outputs/plots/baseline_exact_match_small_multiples.png")
    print("outputs/plots/baseline_token_accuracy_small_multiples.png")
    print("outputs/plots/baseline_generalisation_gap.png")
    print("outputs/plots/baseline_generalisation_gap.csv")
    print("outputs/plots/baseline_task_summary.csv")


if __name__ == "__main__":
    main()