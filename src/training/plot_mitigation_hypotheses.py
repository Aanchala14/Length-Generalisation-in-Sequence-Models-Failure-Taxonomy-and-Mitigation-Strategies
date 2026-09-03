from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MULTISEED_DIR = Path(
    "outputs/results/multiseed_results"
)

MITIGATION_DIR = Path(
    "outputs/results/mitigation_results"
)

OUTPUT_DIR = Path(
    "outputs/plots/final"
)


TRAIN_FIT_THRESHOLD = 90.0

MEANINGFUL_IMPROVEMENT = 1.0

COLLAPSE_THRESHOLD = 10.0


TASK_LABELS = {
    "addition": "Addition",
    "copy": "Delayed Copy",
    "reverse": "Reverse",
}


ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "rope": "RoPE",
}


MITIGATION_STYLES = {
    "baseline": {
        "color": "#1F4E79",
        "marker": "o",
        "linestyle": "-",
    },
    "control": {
        "color": "#626B75",
        "marker": "X",
        "linestyle": "--",
    },
    "mixed": {
        "color": "#4F7FA3",
        "marker": "s",
        "linestyle": "--",
    },
    "mixed_v2": {
        "color": "#7899B0",
        "marker": "P",
        "linestyle": ":",
    },
    "curriculum": {
        "color": "#2E6F75",
        "marker": "^",
        "linestyle": "-.",
    },
    "randomised": {
        "color": "#6B5F7A",
        "marker": "D",
        "linestyle": ":",
    },
}


MITIGATION_COMPARISONS = [
    {
        "task": "addition",
        "encoding": "learned",
        "seed": 42,
        "output_stem": (
            "addition_learned_"
            "baseline_vs_mitigation"
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "role": "Reference baseline",
                "hypothesis": (
                    "Provides the original single-length training "
                    "reference; no mitigation is applied."
                ),
                "path": (
                    MULTISEED_DIR
                    / "addition_train16_learned_seed42_results.csv"
                ),
            },
            {
                "label": "Mixed-length (v1)",
                "style": "mixed",
                "role": "Mitigation",
                "hypothesis": (
                    "Tests whether exposure to operand lengths "
                    "4, 8, 12 and 16 reduces over-specialisation "
                    "to the single training length of 16."
                ),
                "path": (
                    MITIGATION_DIR
                    / "addition_train16_mixed_learned_seed42_results.csv"
                ),
            },
            {
                "label": "Mixed-length (v2)",
                "style": "mixed_v2",
                "role": "Strengthened mitigation",
                "hypothesis": (
                    "Repeats mixed-length exposure with more "
                    "training samples and epochs to test whether "
                    "the first run was limited by data volume "
                    "or optimisation time."
                ),
                "path": (
                    MITIGATION_DIR
                    / "addition_train16_mixed_learned_seed42_v2_results.csv"
                ),
            },
        ],
    },
    {
        "task": "copy",
        "encoding": "sinusoidal",
        "seed": 42,
        "output_stem": (
            "copy_sinusoidal_"
            "baseline_vs_mitigation"
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "role": "Reference baseline",
                "hypothesis": (
                    "Provides the original single-length training "
                    "reference; no mitigation is applied."
                ),
                "path": (
                    MULTISEED_DIR
                    / "copy_train128_sinusoidal_seed42_results.csv"
                ),
            },
            {
                "label": "Single-length control",
                "style": "control",
                "role": "Experimental control",
                "hypothesis": (
                    "Tests whether the mitigation pipeline "
                    "reproduces the original baseline when the "
                    "training-length distribution is unchanged."
                ),
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_single_sinusoidal_seed42_control_results.csv"
                ),
            },
            {
                "label": "Mixed-length",
                "style": "mixed",
                "role": "Mitigation",
                "hypothesis": (
                    "Tests whether exposure to sequence lengths "
                    "32, 64, 96 and 128 reduces over-specialisation "
                    "to the single training length of 128."
                ),
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_mixed_sinusoidal_seed42_results.csv"
                ),
            },
            {
                "label": "Curriculum",
                "style": "curriculum",
                "role": "Mitigation",
                "hypothesis": (
                    "Tests whether sequential training at lengths "
                    "32, 64, 96 and 128 removes an optimisation "
                    "barrier caused by training directly at "
                    "length 128."
                ),
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_curriculum_sinusoidal_seed42_stage4_results.csv"
                ),
            },
            {
                "label": "Randomised padded",
                "style": "randomised",
                "role": "Mitigation",
                "hypothesis": (
                    "Tests whether varying sequence length inside "
                    "a jointly padded training set reduces reliance "
                    "on fixed separator and output positions."
                ),
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_randomised_sinusoidal_seed42_results.csv"
                ),
            },
        ],
    },
    {
        "task": "copy",
        "encoding": "rope",
        "seed": 42,
        "output_stem": (
            "copy_rope_"
            "baseline_vs_mitigation"
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "role": "Reference baseline",
                "hypothesis": (
                    "Provides the original single-length training "
                    "reference; no mitigation is applied."
                ),
                "path": (
                    MULTISEED_DIR
                    / "copy_train128_rope_seed42_results.csv"
                ),
            },
            {
                "label": "Curriculum",
                "style": "curriculum",
                "role": "Mitigation",
                "hypothesis": (
                    "Tests whether sequential training at lengths "
                    "32, 64, 96 and 128 removes an optimisation "
                    "barrier caused by training directly at "
                    "length 128."
                ),
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_curriculum_rope_seed42_stage4_results.csv"
                ),
            },
            {
                "label": "Randomised padded",
                "style": "randomised",
                "role": "Mitigation",
                "hypothesis": (
                    "Tests whether varying sequence length inside "
                    "a jointly padded training set reduces reliance "
                    "on fixed separator and output placement."
                ),
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_randomised_rope_seed42_results.csv"
                ),
            },
        ],
    },
    {
        "task": "reverse",
        "encoding": "learned",
        "seed": 42,
        "output_stem": (
            "reverse_learned_"
            "baseline_vs_mitigation"
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "role": "Reference baseline",
                "hypothesis": (
                    "Provides the original single-length training "
                    "reference; no mitigation is applied."
                ),
                "path": (
                    MULTISEED_DIR
                    / "reverse_train128_learned_seed42_results.csv"
                ),
            },
            {
                "label": "Curriculum",
                "style": "curriculum",
                "role": "Mitigation",
                "hypothesis": (
                    "Tests whether sequential training at lengths "
                    "32, 64, 96 and 128 removes an optimisation "
                    "barrier caused by training directly at "
                    "length 128."
                ),
                "path": (
                    MITIGATION_DIR
                    / "reverse_train128_curriculum_learned_seed42_stage4_results.csv"
                ),
            },
        ],
    },
]


def set_report_style():

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "normal",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 8.2,
        "legend.title_fontsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_line_axis(ax):

    ax.set_axisbelow(True)

    ax.grid(
        axis="y",
        color="#D7DCE2",
        linestyle="--",
        linewidth=0.7,
        alpha=0.70
    )


def save_figure(
    fig,
    output_stem
):

    fig.savefig(
        OUTPUT_DIR / f"{output_stem}.png",
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        OUTPUT_DIR / f"{output_stem}.pdf",
        bbox_inches="tight",
        facecolor="white"
    )


def length_positions(lengths):

    return {
        int(length): index
        for index, length in enumerate(
            sorted(lengths)
        )
    }


def add_training_length_marker(
    ax,
    train_position
):

    ax.axvline(
        train_position,
        color="#3F3F3F",
        linestyle="--",
        linewidth=1.1,
        zorder=1
    )

    ax.annotate(
        "training length",
        xy=(
            train_position,
            104
        ),
        xytext=(
            7,
            0
        ),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.2,
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.20",
            "facecolor": "white",
            "edgecolor": "#C8CDD2",
            "alpha": 0.96,
        },
    )


def add_collapse_threshold(ax):

    ax.axhline(
        COLLAPSE_THRESHOLD,
        color="#666666",
        linestyle=":",
        linewidth=1.1,
        alpha=0.90,
        zorder=1
    )

    ax.annotate(
        (
            f"{COLLAPSE_THRESHOLD:.0f}% "
            "collapse threshold"
        ),
        xy=(
            0.98,
            0.15
        ),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=8.2,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
    )


def load_raw_experiment(
    path,
    expected_task,
    expected_encoding,
    expected_seed
):

    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw experiment file: {path}"
        )

    frame = pd.read_csv(
        path
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

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    tasks = set(
        frame["Task"].unique()
    )

    encodings = set(
        frame["Positional Encoding"].unique()
    )

    seeds = set(
        frame["Seed"].astype(int).unique()
    )

    if tasks != {expected_task}:
        raise ValueError(
            f"Task mismatch in {path}: "
            f"expected {expected_task}, found {tasks}"
        )

    if encodings != {expected_encoding}:
        raise ValueError(
            f"Encoding mismatch in {path}: "
            f"expected {expected_encoding}, "
            f"found {encodings}"
        )

    if seeds != {expected_seed}:
        raise ValueError(
            f"Seed mismatch in {path}: "
            f"expected {expected_seed}, found {seeds}"
        )

    frame = frame.copy()

    numeric_columns = [
        "Train Length",
        "Test Length",
        "Seed",
        "Token Accuracy",
        "Exact Match Accuracy",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise"
        )

    duplicate_lengths = frame[
        "Test Length"
    ].duplicated()

    if duplicate_lengths.any():
        duplicates = (
            frame.loc[
                duplicate_lengths,
                "Test Length"
            ]
            .astype(int)
            .tolist()
        )

        raise ValueError(
            f"{path} contains duplicate test lengths: "
            f"{duplicates}"
        )

    frame["Token Error Rate"] = (
        100.0 - frame["Token Accuracy"]
    )

    frame["Exact Match Error Rate"] = (
        100.0 - frame["Exact Match Accuracy"]
    )

    return frame.sort_values(
        "Test Length"
    ).reset_index(
        drop=True
    )


def get_exact_match_at_length(
    frame,
    test_length,
    series_label
):

    row = frame[
        frame["Test Length"] == test_length
    ]

    if row.empty:
        raise ValueError(
            f"{series_label} is missing test length "
            f"{test_length}."
        )

    return float(
        row["Exact Match Accuracy"].iloc[0]
    )


def analyse_condition(
    item,
    baseline_frame,
    train_length,
    first_unseen_length,
    longest_test_length,
    baseline_first_unseen_exact
):

    frame = item["frame"]

    exact_at_train = get_exact_match_at_length(
        frame,
        train_length,
        item["label"]
    )

    first_unseen_exact = get_exact_match_at_length(
        frame,
        first_unseen_length,
        item["label"]
    )

    longest_exact = get_exact_match_at_length(
        frame,
        longest_test_length,
        item["label"]
    )

    if exact_at_train < TRAIN_FIT_THRESHOLD:
        training_status = "Training underfit"
    else:
        training_status = "Training task learned"

    outcome = (
        f"Training exact match = {exact_at_train:.2f}%; "
        f"first-unseen exact match at length "
        f"{first_unseen_length} = "
        f"{first_unseen_exact:.2f}%; "
        f"longest-length exact match at length "
        f"{longest_test_length} = "
        f"{longest_exact:.2f}%."
    )

    role = item["role"]

    if role == "Reference baseline":
        interpretation = "Reference failure profile"

    elif role == "Experimental control":
        comparison_frame = frame[[
            "Test Length",
            "Exact Match Accuracy",
        ]].merge(
            baseline_frame[[
                "Test Length",
                "Exact Match Accuracy",
            ]],
            on="Test Length",
            suffixes=(
                "_control",
                "_baseline"
            ),
            how="inner"
        )

        if comparison_frame.empty:
            raise ValueError(
                "The control and baseline do not share "
                "any test lengths."
            )

        maximum_control_gap = float(
            np.max(
                np.abs(
                    comparison_frame[
                        "Exact Match Accuracy_control"
                    ].to_numpy()
                    - comparison_frame[
                        "Exact Match Accuracy_baseline"
                    ].to_numpy()
                )
            )
        )

        outcome += (
            " Maximum exact-match difference from the "
            f"baseline = {maximum_control_gap:.2f} "
            "percentage points."
        )

        if (
            training_status == "Training task learned"
            and maximum_control_gap
            <= MEANINGFUL_IMPROVEMENT
        ):
            interpretation = (
                "Control reproduced baseline"
            )
        else:
            interpretation = (
                "Control did not reproduce baseline"
            )

    elif training_status == "Training underfit":
        interpretation = (
            "Inconclusive: training underfit"
        )

        outcome += (
            " Because training-length exact match is "
            f"below the {TRAIN_FIT_THRESHOLD:.0f}% "
            "eligibility threshold, extrapolation "
            "cannot be used to judge this mitigation."
        )

    elif first_unseen_exact >= TRAIN_FIT_THRESHOLD:
        interpretation = (
            "Successful extrapolation recovery"
        )

        outcome += (
            " The mitigation retained training-level "
            "performance at the first unseen length."
        )

    elif (
        first_unseen_exact
        > baseline_first_unseen_exact
        + MEANINGFUL_IMPROVEMENT
    ):
        improvement = (
            first_unseen_exact
            - baseline_first_unseen_exact
        )

        interpretation = (
            "Partial improvement below recovery criterion"
        )

        outcome += (
            " The first-unseen result improves on the "
            f"baseline by {improvement:.2f} percentage "
            "points, but does not reach the "
            f"{TRAIN_FIT_THRESHOLD:.0f}% recovery criterion."
        )

    else:
        interpretation = (
            "Valid negative result: extrapolation unchanged"
        )

        outcome += (
            " The training task was learned, but there "
            "was no meaningful first-unseen improvement "
            "over the baseline "
            f"({baseline_first_unseen_exact:.2f}%)."
        )

    return {
        "training_status": training_status,
        "exact_at_train": exact_at_train,
        "first_unseen_exact": first_unseen_exact,
        "longest_exact": longest_exact,
        "outcome": outcome,
        "interpretation": interpretation,
    }


def build_legend_label(
    label,
    training_status
):

    if training_status == "Training underfit":
        return (
            f"{label} (training underfit)"
        )

    return label


def plot_comparison(
    comparison,
    plot_series,
    train_length
):

    task = comparison["task"]
    encoding = comparison["encoding"]
    seed = comparison["seed"]

    all_lengths = sorted({
        int(length)
        for item in plot_series
        for length in item["frame"]["Test Length"]
    })

    positions = length_positions(
        all_lengths
    )

    number_of_series = len(
        plot_series
    )

    if number_of_series == 1:
        marker_offsets = np.array([
            0.0
        ])
    else:
        marker_offsets = np.linspace(
            -0.10,
            0.10,
            number_of_series
        )

    fig, ax = plt.subplots(
        figsize=(7.4, 4.6)
    )

    style_line_axis(
        ax
    )

    add_training_length_marker(
        ax,
        positions[train_length]
    )

    add_collapse_threshold(
        ax
    )

    legend_handles = []

    for series_index, item in enumerate(
        plot_series
    ):
        frame = item["frame"]

        style = MITIGATION_STYLES[
            item["style"]
        ]

        true_x_values = [
            positions[int(length)]
            for length in frame["Test Length"]
        ]

        marker_x_values = [
            positions[int(length)]
            + marker_offsets[series_index]
            for length in frame["Test Length"]
        ]

        y_values = frame[
            "Exact Match Accuracy"
        ].to_numpy()

        ax.plot(
            true_x_values,
            y_values,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.1,
            alpha=0.95,
            zorder=2 + series_index,
        )

        ax.scatter(
            marker_x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            s=42,
            edgecolors="white",
            linewidths=0.65,
            alpha=0.98,
            zorder=5 + series_index,
        )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.1,
                markersize=5.8,
                markeredgecolor="white",
                markeredgewidth=0.6,
                label=item["legend_label"],
            )
        )

    task_label = TASK_LABELS[
        task
    ]

    encoding_label = ENCODING_LABELS[
        encoding
    ]

    ax.set_title(
        f"{task_label} ({encoding_label}): "
        "baseline vs mitigation"
    )

    ax.set_xlabel(
        "Test sequence length"
    )

    ax.set_ylabel(
        "Exact-match accuracy (%)"
    )

    ax.set_ylim(
        -4,
        110
    )

    ax.set_xlim(
        -0.45,
        len(all_lengths) - 1 + 0.45
    )

    ax.set_xticks(
        range(len(all_lengths))
    )

    ax.set_xticklabels([
        str(length)
        for length in all_lengths
    ])

    ax.legend(
        handles=legend_handles,
        title=(
            "Training condition\n"
            f"Seed {seed}; markers offset"
        ),
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#C8CDD2"
    )

    fig.tight_layout()

    save_figure(
        fig,
        comparison["output_stem"]
    )

    plt.close(
        fig
    )


def process_comparison(
    comparison
):

    task = comparison["task"]
    encoding = comparison["encoding"]
    seed = comparison["seed"]

    plot_series = []

    train_lengths = set()

    for series in comparison["series"]:
        frame = load_raw_experiment(
            path=series["path"],
            expected_task=task,
            expected_encoding=encoding,
            expected_seed=seed
        )

        train_length = int(
            frame["Train Length"].iloc[0]
        )

        train_lengths.add(
            train_length
        )

        plot_series.append({
            "label": series["label"],
            "style": series["style"],
            "role": series["role"],
            "hypothesis": series["hypothesis"],
            "source_path": series["path"],
            "frame": frame,
        })

    if len(train_lengths) != 1:
        raise ValueError(
            "Matched mitigation comparison contains "
            f"different training lengths: {train_lengths}"
        )

    train_length = next(
        iter(train_lengths)
    )

    baseline_items = [
        item
        for item in plot_series
        if item["role"] == "Reference baseline"
    ]

    if len(baseline_items) != 1:
        raise ValueError(
            "Each comparison must contain exactly one "
            "reference baseline."
        )

    baseline_item = baseline_items[0]

    baseline_frame = baseline_item[
        "frame"
    ]

    baseline_unseen = baseline_frame[
        baseline_frame["Test Length"]
        > train_length
    ].sort_values(
        "Test Length"
    )

    if baseline_unseen.empty:
        raise ValueError(
            "The reference baseline has no unseen "
            "test length."
        )

    first_unseen_length = int(
        baseline_unseen[
            "Test Length"
        ].iloc[0]
    )

    longest_test_length = int(
        baseline_unseen[
            "Test Length"
        ].iloc[-1]
    )

    baseline_first_unseen_exact = float(
        baseline_unseen[
            "Exact Match Accuracy"
        ].iloc[0]
    )

    detailed_rows = []

    summary_rows = []

    for item in plot_series:
        analysis = analyse_condition(
            item=item,
            baseline_frame=baseline_frame,
            train_length=train_length,
            first_unseen_length=first_unseen_length,
            longest_test_length=longest_test_length,
            baseline_first_unseen_exact=(
                baseline_first_unseen_exact
            )
        )

        item.update(
            analysis
        )

        item["legend_label"] = build_legend_label(
            item["label"],
            item["training_status"]
        )

        summary_rows.append({
            "Task": TASK_LABELS[task],
            "Positional Encoding": (
                ENCODING_LABELS[encoding]
            ),
            "Seed": seed,
            "Training Condition": item["label"],
            "Experimental Role": item["role"],
            "Specific Hypothesis": (
                item["hypothesis"]
            ),
            "Training Status": (
                item["training_status"]
            ),
            "Training Exact Match (%)": round(
                item["exact_at_train"],
                4
            ),
            "First Unseen Length": (
                first_unseen_length
            ),
            "First-Unseen Exact Match (%)": round(
                item["first_unseen_exact"],
                4
            ),
            "Longest Test Length": (
                longest_test_length
            ),
            "Longest-Test Exact Match (%)": round(
                item["longest_exact"],
                4
            ),
            "Observed Outcome": item["outcome"],
            "Interpretation": (
                item["interpretation"]
            ),
            "Source CSV": str(
                item["source_path"]
            ),
        })

        for _, row in item["frame"].iterrows():
            detailed_rows.append({
                "Task": task,
                "Task Label": TASK_LABELS[task],
                "Positional Encoding": encoding,
                "Positional Encoding Label": (
                    ENCODING_LABELS[encoding]
                ),
                "Training Condition": (
                    item["label"]
                ),
                "Experimental Role": (
                    item["role"]
                ),
                "Training Status": (
                    item["training_status"]
                ),
                "Seed": int(
                    row["Seed"]
                ),
                "Train Length": int(
                    row["Train Length"]
                ),
                "Test Length": int(
                    row["Test Length"]
                ),
                "Token Accuracy (%)": (
                    row["Token Accuracy"]
                ),
                "Token Error Rate (%)": (
                    row["Token Error Rate"]
                ),
                "Exact-Match Accuracy (%)": (
                    row["Exact Match Accuracy"]
                ),
                "Exact-Match Error Rate (%)": (
                    row["Exact Match Error Rate"]
                ),
                "Specific Hypothesis": (
                    item["hypothesis"]
                ),
                "Observed Outcome": (
                    item["outcome"]
                ),
                "Interpretation": (
                    item["interpretation"]
                ),
                "Source CSV": str(
                    item["source_path"]
                ),
            })

    plot_comparison(
        comparison=comparison,
        plot_series=plot_series,
        train_length=train_length
    )

    detailed_data = pd.DataFrame(
        detailed_rows
    )

    detailed_data.to_csv(
        OUTPUT_DIR
        / (
            f"{comparison['output_stem']}"
            "_data.csv"
        ),
        index=False
    )

    return summary_rows


def main():

    set_report_style()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_summary_rows = []

    for comparison in MITIGATION_COMPARISONS:
        comparison_rows = process_comparison(
            comparison
        )

        all_summary_rows.extend(
            comparison_rows
        )

    summary = pd.DataFrame(
        all_summary_rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "mitigation_hypothesis_summary.csv",
        index=False
    )

    print(
        "Saved mitigation PNG/PDF figures, detailed "
        "plot-data CSVs and mitigation_hypothesis_summary.csv "
        f"to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()