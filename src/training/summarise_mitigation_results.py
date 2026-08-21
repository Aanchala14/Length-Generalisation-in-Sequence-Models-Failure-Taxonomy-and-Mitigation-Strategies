from pathlib import Path

import pandas as pd


BASELINE_DIR = Path("outputs/results/multiseed_results")
MITIGATION_DIR = Path("outputs/results/mitigation_results")
OUTPUT_DIR = Path("outputs/analysis/mitigation_summary")


MITIGATION_FILES = {
    "Copy Baseline Sinusoidal": BASELINE_DIR / "copy_train128_sinusoidal_seed42_results.csv",
    "Copy Baseline RoPE": BASELINE_DIR / "copy_train128_rope_seed42_results.csv",
    "Copy Mixed Sinusoidal": MITIGATION_DIR / "copy_train128_mixed_sinusoidal_seed42_results.csv",
    "Copy Single Control Sinusoidal": MITIGATION_DIR / "copy_train128_single_sinusoidal_seed42_control_results.csv",
    "Copy Curriculum Sinusoidal": MITIGATION_DIR / "copy_train128_curriculum_sinusoidal_seed42_stage4_results.csv",
    "Copy Curriculum RoPE": MITIGATION_DIR / "copy_train128_curriculum_rope_seed42_stage4_results.csv",
    "Copy Randomised Sinusoidal": MITIGATION_DIR / "copy_train128_randomised_sinusoidal_seed42_results.csv",
    "Copy Randomised RoPE": MITIGATION_DIR / "copy_train128_randomised_rope_seed42_results.csv",
    "Reverse Baseline Learned": BASELINE_DIR / "reverse_train128_learned_seed42_results.csv",
    "Reverse Curriculum Learned": MITIGATION_DIR / "reverse_train128_curriculum_learned_seed42_stage4_results.csv",
    "Addition Baseline Learned": BASELINE_DIR / "addition_train16_learned_seed42_results.csv",
    "Addition Mixed Learned": MITIGATION_DIR / "addition_train16_mixed_learned_seed42_results.csv",
    "Addition Mixed Learned V2": MITIGATION_DIR / "addition_train16_mixed_learned_seed42_v2_results.csv",
}


def load_result(label, path):
    if not path.exists():
        print(f"Skipping missing file: {path}")
        return None

    frame = pd.read_csv(path)
    frame["Experiment"] = label

    return frame


def get_row(frame, test_length):
    row = frame[
        frame["Test Length"] == test_length
    ]

    if row.empty:
        return None

    return row.iloc[0]


def summarise_experiment(label, frame):
    task = frame["Task"].iloc[0]
    train_length = int(frame["Train Length"].iloc[0])
    positional_encoding = frame["Positional Encoding"].iloc[0]
    seed = int(frame["Seed"].iloc[0])

    train_row = get_row(frame, train_length)
    longest_row = frame.sort_values("Test Length").iloc[-1]

    row_256 = get_row(frame, 256)
    row_512 = get_row(frame, 512)
    row_1024 = get_row(frame, 1024)

    exact_train = train_row["Exact Match Accuracy"]
    exact_longest = longest_row["Exact Match Accuracy"]

    if exact_train < 10:
        conclusion = "Underfits training length"
    elif exact_longest < 10:
        conclusion = "Fits training length; fails extrapolation"
    else:
        conclusion = "Improves extrapolation"

    return {
        "Experiment": label,
        "Task": task,
        "Train Length": train_length,
        "Positional Encoding": positional_encoding,
        "Seed": seed,
        "Exact@Train": exact_train,
        "Exact@256": row_256["Exact Match Accuracy"] if row_256 is not None else "",
        "Exact@512": row_512["Exact Match Accuracy"] if row_512 is not None else "",
        "Exact@1024": row_1024["Exact Match Accuracy"] if row_1024 is not None else "",
        "Token@Train": train_row["Token Accuracy"],
        "Token@1024": row_1024["Token Accuracy"] if row_1024 is not None else "",
        "Generalisation Gap": exact_train - exact_longest,
        "Conclusion": conclusion,
    }


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for label, path in MITIGATION_FILES.items():
        frame = load_result(label, path)

        if frame is None:
            continue

        rows.append(
            summarise_experiment(label, frame)
        )

    summary = pd.DataFrame(rows)

    summary.to_csv(
        OUTPUT_DIR / "mitigation_summary.csv",
        index=False
    )

    print(
        summary.to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_DIR / 'mitigation_summary.csv'}"
    )


if __name__ == "__main__":
    main()