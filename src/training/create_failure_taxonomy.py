from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs/analysis/failure_taxonomy")


ROWS = [
    {
        "Failure Type": "Extrapolation collapse",
        "Definition": "The model fits the training length but exact-match accuracy collapses outside the training range.",
        "Evidence": "High Exact@Train but 0% Exact@Long in baseline and PE experiments.",
        "Tasks Affected": "Addition, delayed copy, reverse",
        "Example": "Addition learned: Exact@16 = 94.1%, Exact@32 = 0%, Exact@1024 = 0%.",
        "Interpretation": "The model learns a length-specific solution rather than a length-general algorithm.",
    },
    {
        "Failure Type": "Training-length underfitting",
        "Definition": "The model fails even at the training length.",
        "Evidence": "Exact@Train remains near 0% for some positional encodings and mitigation variants.",
        "Tasks Affected": "Addition, delayed copy, reverse",
        "Example": "NoPE and ALiBi often fail at the training length; mixed-length addition also gives Exact@16 = 0%.",
        "Interpretation": "Some configurations are not valid extrapolation tests because they do not learn the in-distribution task.",
    },
    {
        "Failure Type": "Seed instability",
        "Definition": "The same task and positional encoding produce different train-length outcomes across random seeds.",
        "Evidence": "Large standard deviation and standard error in the multi-seed summary.",
        "Tasks Affected": "Delayed copy learned, reverse sinusoidal",
        "Example": "Delayed copy learned has low mean Exact@Train with high variance across seeds.",
        "Interpretation": "Some results depend strongly on optimisation path, so multi-seed reporting is necessary.",
    },
    {
        "Failure Type": "Attention diffusion",
        "Definition": "Attention becomes increasingly diffuse or less locally structured as sequence length grows.",
        "Evidence": "Normalised attention entropy increases and local attention ratio decreases at failure lengths.",
        "Tasks Affected": "Addition, delayed copy, reverse",
        "Example": "Reverse learned normalised entropy increases from train length to 1024 while exact match falls to 0%.",
        "Interpretation": "The attention pattern used at training length does not transfer reliably to longer contexts.",
    },
    {
        "Failure Type": "Mitigation-resistant failure",
        "Definition": "Simple mitigation strategies preserve or recover train-length accuracy but do not improve extrapolation.",
        "Evidence": "Curriculum and randomised padded training keep Exact@Train high but Exact@256+ remains 0%.",
        "Tasks Affected": "Delayed copy, reverse",
        "Example": "Copy randomised RoPE: Exact@128 = 100%, Exact@256/512/1024 = 0%.",
        "Interpretation": "Exposure to multiple shorter lengths is insufficient to force algorithmic length generalisation.",
    },
]


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    taxonomy = pd.DataFrame(ROWS)

    output_path = OUTPUT_DIR / "failure_taxonomy.csv"

    taxonomy.to_csv(
        output_path,
        index=False
    )

    print(taxonomy.to_string(index=False))
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()