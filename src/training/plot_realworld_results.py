import os
import glob
import math
import csv

import matplotlib.pyplot as plt


RESULTS_DIR = "outputs/results/realworld"
OUTPUT_DIR = "outputs/plots/realworld"


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) <= 1:
        return 0.0

    value_mean = mean(values)

    variance = sum(
        (value - value_mean) ** 2
        for value in values
    ) / (len(values) - 1)

    return math.sqrt(variance)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grouped = {}

    for path in glob.glob(f"{RESULTS_DIR}/*.csv"):
        with open(path, newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                positional_encoding = row["Positional Encoding"]
                test_length = int(row["Test Length"])
                accuracy = float(row["Accuracy"])

                key = (
                    positional_encoding,
                    test_length
                )

                grouped.setdefault(key, []).append(accuracy)

    summary_rows = []

    positional_encodings = sorted(
        set(key[0] for key in grouped)
    )

    test_lengths = sorted(
        set(key[1] for key in grouped)
    )

    for positional_encoding in positional_encodings:
        for test_length in test_lengths:
            values = grouped[
                (
                    positional_encoding,
                    test_length
                )
            ]

            summary_rows.append({
                "Dataset": "SelfRegulationSCP2",
                "Train Length": 256,
                "Test Length": test_length,
                "Positional Encoding": positional_encoding,
                "Seeds": len(values),
                "Mean Accuracy": mean(values),
                "Std Accuracy": std(values),
                "SE Accuracy": std(values) / math.sqrt(len(values)),
            })

    summary_path = f"{OUTPUT_DIR}/selfregulationscp2_summary.csv"

    with open(summary_path, "w", newline="") as file:
        fieldnames = [
            "Dataset",
            "Train Length",
            "Test Length",
            "Positional Encoding",
            "Seeds",
            "Mean Accuracy",
            "Std Accuracy",
            "SE Accuracy",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    plt.figure(figsize=(9, 5.5))

    for positional_encoding in positional_encodings:
        means = []
        errors = []

        for test_length in test_lengths:
            values = grouped[
                (
                    positional_encoding,
                    test_length
                )
            ]

            means.append(mean(values))
            errors.append(std(values) / math.sqrt(len(values)))

        plt.errorbar(
            test_lengths,
            means,
            yerr=errors,
            marker="o",
            linewidth=2,
            capsize=4,
            label=positional_encoding
        )

    plt.xlabel("Test sequence length")
    plt.ylabel("Classification accuracy (%)")
    plt.title(
        "Real-world extension: SelfRegulationSCP2 accuracy vs length"
    )
    plt.xticks(test_lengths)
    plt.ylim(40, 65)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Positional encoding")
    plt.tight_layout()

    plot_path = (
        f"{OUTPUT_DIR}/"
        "selfregulationscp2_accuracy_mean_se.png"
    )

    plt.savefig(plot_path, dpi=300)

    print(f"Saved summary to: {summary_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()