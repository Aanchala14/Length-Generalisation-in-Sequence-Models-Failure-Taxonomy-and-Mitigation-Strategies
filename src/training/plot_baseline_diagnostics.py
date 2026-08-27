from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "outputs/results/multiseed_results"
)

OUTPUT_DIR = Path(
    "outputs/plots/final"
)


TRAIN_FIT_THRESHOLD = 90.0

COLLAPSE_THRESHOLD = 10.0


TASK_LABELS = {
    "addition": "Addition",
    "copy": "Delayed Copy",
    "reverse": "Reverse",
}

TASK_ORDER = [
    "addition",
    "copy",
    "reverse",
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


# Same restrained palette used in the final-report script.
ENCODING_COLORS = {
    "learned": "#1F4E79",
    "sinusoidal": "#2E6F75",
    "none": "#707070",
    "alibi": "#4F7FA3",
    "rope": "#6B5F7A",
}

ENCODING_MARKERS = {
    "learned": "o",
    "sinusoidal": "s",
    "none": "D",
    "alibi": "^",
    "rope": "P",
}

ENCODING_LINESTYLES = {
    "learned": "-",
    "sinusoidal": "--",
    "none": "-.",
    "alibi": ":",
    "rope": (0, (4, 1, 1, 1)),
}


def set_report_style():
    """
    Apply the same style used by all final thesis figures.
    """

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
        "legend.fontsize": 8.5,
        "legend.title_fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax):
    """
    Apply consistent horizontal grid lines.
    """

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
    """
    Save both a PNG and a vector PDF.
    """

    png_path = OUTPUT_DIR / f"{output_stem}.png"

    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"

    fig.savefig(
        png_path,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white"
    )


def load_results():
    """
    Load all baseline result files.
    """

    files = sorted(
        RESULTS_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No result CSV files found in {RESULTS_DIR}"
        )

    results = pd.concat(
        [
            pd.read_csv(file)
            for file in files
        ],
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

    missing_columns = (
        required_columns - set(results.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing result columns: "
            f"{sorted(missing_columns)}"
        )

    results["Token Error Rate"] = (
        100.0 - results["Token Accuracy"]
    )

    results["Exact Match Error Rate"] = (
        100.0 - results["Exact Match Accuracy"]
    )

    return results


def build_baseline_eligibility(results):
    """
    Determine whether each task and encoding learns the
    training-length task consistently across all seeds.
    """

    rows = []

    for task in TASK_ORDER:
        for encoding in ENCODING_ORDER:
            frame = results[
                (
                    results["Task"] == task
                )
                & (
                    results["Positional Encoding"]
                    == encoding
                )
            ].copy()

            if frame.empty:
                continue

            train_length = int(
                frame["Train Length"].iloc[0]
            )

            train_rows = frame[
                frame["Test Length"]
                == train_length
            ].sort_values(
                "Seed"
            )

            if train_rows.empty:
                continue

            total_seeds = int(
                train_rows["Seed"].nunique()
            )

            successful_mask = (
                train_rows["Exact Match Accuracy"]
                >= TRAIN_FIT_THRESHOLD
            )

            successful_seeds = (
                train_rows.loc[
                    successful_mask,
                    "Seed"
                ]
                .astype(int)
                .tolist()
            )

            successful_count = len(
                successful_seeds
            )

            train_mean = train_rows[
                "Exact Match Accuracy"
            ].mean()

            train_std = train_rows[
                "Exact Match Accuracy"
            ].std()

            if pd.isna(train_std):
                train_std = 0.0

            train_min = train_rows[
                "Exact Match Accuracy"
            ].min()

            train_max = train_rows[
                "Exact Match Accuracy"
            ].max()

            if successful_count == total_seeds:
                status = "Eligible"

                interpretation = (
                    "Stable training-length fit"
                )

            elif successful_count == 0:
                status = "Not eligible"

                if train_mean < COLLAPSE_THRESHOLD:
                    interpretation = (
                        "Training-length underfitting"
                    )
                else:
                    interpretation = (
                        "Partial training fit"
                    )

            else:
                status = "Seed unstable"

                interpretation = (
                    "Successful and unsuccessful seeds"
                )

            rows.append({
                "Task": task,
                "Task Label": TASK_LABELS[task],
                "Positional Encoding": encoding,
                "Encoding Label": (
                    ENCODING_LABELS[encoding]
                ),
                "Train Length": train_length,
                "Total Seeds": total_seeds,
                "Successful Seeds": successful_count,
                "Successful Seed IDs": (
                    ", ".join(
                        str(seed)
                        for seed in successful_seeds
                    )
                    if successful_seeds
                    else "None"
                ),
                "Exact@Train Mean (%)": round(
                    train_mean,
                    2
                ),
                "Exact@Train SD "
                "(percentage points)": round(
                    train_std,
                    2
                ),
                "Exact@Train Minimum (%)": round(
                    train_min,
                    2
                ),
                "Exact@Train Maximum (%)": round(
                    train_max,
                    2
                ),
                "Eligibility": status,
                "Interpretation": interpretation,
            })

    return pd.DataFrame(
        rows
    )


def length_positions(lengths):
    """
    Map lengths to equally spaced display positions.
    """

    return {
        length: index
        for index, length in enumerate(
            sorted(lengths)
        )
    }


def first_collapse_length(
    aggregate,
    train_length
):
    """
    Find the first unseen length with mean exact-match
    accuracy below the collapse threshold.
    """

    extrapolation = aggregate[
        aggregate["Test Length"]
        > train_length
    ].sort_values(
        "Test Length"
    )

    collapsed = extrapolation[
        extrapolation["Exact Mean"]
        < COLLAPSE_THRESHOLD
    ]

    if collapsed.empty:
        return None

    return int(
        collapsed["Test Length"].iloc[0]
    )


def aggregate_encoding(frame):
    """
    Aggregate one task and encoding across seeds.
    """

    aggregate = frame.groupby(
        "Test Length",
        as_index=False
    ).agg(
        Seed_Count=(
            "Seed",
            "nunique"
        ),
        Exact_Mean=(
            "Exact Match Accuracy",
            "mean"
        ),
        Exact_SD=(
            "Exact Match Accuracy",
            "std"
        ),
        Exact_Minimum=(
            "Exact Match Accuracy",
            "min"
        ),
        Exact_Maximum=(
            "Exact Match Accuracy",
            "max"
        ),
        Token_Mean=(
            "Token Accuracy",
            "mean"
        ),
        Token_SD=(
            "Token Accuracy",
            "std"
        ),
    )

    aggregate[
        "Exact_SD"
    ] = aggregate[
        "Exact_SD"
    ].fillna(
        0.0
    )

    aggregate[
        "Token_SD"
    ] = aggregate[
        "Token_SD"
    ].fillna(
        0.0
    )

    aggregate = aggregate.rename(
        columns={
            "Seed_Count": "Seed Count",
            "Exact_Mean": "Exact Mean",
            "Exact_SD": "Exact SD",
            "Exact_Minimum": "Exact Minimum",
            "Exact_Maximum": "Exact Maximum",
            "Token_Mean": "Token Mean",
            "Token_SD": "Token SD",
        }
    )

    return aggregate.sort_values(
        "Test Length"
    )


def add_training_length_marker(
    ax,
    train_position
):
    """
    Mark the exact training length.
    """

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
    """
    Add the operational collapse threshold.
    """

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


def save_training_fit_by_seed(
    results,
    eligibility
):
    """
    Show every seed at the training length.

    This makes underfitting and seed instability visible
    instead of hiding them inside a mean and error bar.
    """

    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(TASK_ORDER),
        figsize=(12.2, 4.7),
        sharey=True
    )

    seed_markers = {
        42: "o",
        123: "s",
        2024: "^",
    }

    for axis_index, task in enumerate(
        TASK_ORDER
    ):
        ax = axes[
            axis_index
        ]

        style_axis(
            ax
        )

        task_results = results[
            results["Task"] == task
        ].copy()

        task_eligibility = eligibility[
            eligibility["Task"] == task
        ].copy()

        available_encodings = [
            encoding
            for encoding in ENCODING_ORDER
            if encoding
            in task_results[
                "Positional Encoding"
            ].unique()
        ]

        x_positions = {
            encoding: index
            for index, encoding
            in enumerate(available_encodings)
        }

        for encoding in available_encodings:
            frame = task_results[
                task_results[
                    "Positional Encoding"
                ]
                == encoding
            ].copy()

            train_length = int(
                frame["Train Length"].iloc[0]
            )

            train_rows = frame[
                frame["Test Length"]
                == train_length
            ].sort_values(
                "Seed"
            )

            x_position = x_positions[
                encoding
            ]

            seed_values = sorted(
                train_rows["Seed"].astype(int).unique()
            )

            if len(seed_values) == 1:
                offsets = {
                    seed_values[0]: 0.0
                }
            else:
                offset_values = np.linspace(
                    -0.13,
                    0.13,
                    len(seed_values)
                )

                offsets = {
                    seed: offset
                    for seed, offset
                    in zip(
                        seed_values,
                        offset_values
                    )
                }

            for _, row in train_rows.iterrows():
                seed = int(
                    row["Seed"]
                )

                ax.scatter(
                    x_position + offsets[seed],
                    row["Exact Match Accuracy"],
                    color=ENCODING_COLORS[
                        encoding
                    ],
                    marker=seed_markers.get(
                        seed,
                        "o"
                    ),
                    s=48,
                    edgecolors="white",
                    linewidths=0.65,
                    zorder=4
                )

            mean_value = train_rows[
                "Exact Match Accuracy"
            ].mean()

            ax.scatter(
                x_position,
                mean_value,
                color="#202020",
                marker="_",
                s=360,
                linewidths=2.0,
                zorder=5
            )

            eligibility_row = task_eligibility[
                task_eligibility[
                    "Positional Encoding"
                ]
                == encoding
            ]

            if not eligibility_row.empty:
                successful = int(
                    eligibility_row[
                        "Successful Seeds"
                    ].iloc[0]
                )

                total = int(
                    eligibility_row[
                        "Total Seeds"
                    ].iloc[0]
                )

                annotation_y = min(
                    train_rows[
                        "Exact Match Accuracy"
                    ].max() + 4.0,
                    106.0
                )

                ax.text(
                    x_position,
                    annotation_y,
                    f"{successful}/{total} fit",
                    ha="center",
                    va="bottom",
                    fontsize=7.6,
                    color="#444444"
                )

        ax.axhline(
            TRAIN_FIT_THRESHOLD,
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            alpha=0.90
        )

        ax.set_title(
            TASK_LABELS[task]
        )

        ax.set_xlabel(
            "Positional encoding"
        )

        ax.set_xticks(
            range(len(available_encodings))
        )

        ax.set_xticklabels(
            [
                ENCODING_LABELS[encoding]
                for encoding in available_encodings
            ],
            rotation=32,
            ha="right"
        )

        ax.set_ylim(
            -4,
            110
        )

        ax.set_xlim(
            -0.45,
            len(available_encodings) - 1 + 0.45
        )

    axes[0].set_ylabel(
        "Training-length exact-match accuracy (%)"
    )

    seed_handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="o",
            linestyle="none",
            markersize=6,
            markeredgecolor="white",
            label="Seed 42"
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="s",
            linestyle="none",
            markersize=6,
            markeredgecolor="white",
            label="Seed 123"
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            marker="^",
            linestyle="none",
            markersize=6,
            markeredgecolor="white",
            label="Seed 2024"
        ),
        Line2D(
            [0],
            [0],
            color="#202020",
            marker="_",
            linestyle="none",
            markersize=13,
            markeredgewidth=2,
            label="Mean"
        ),
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle="--",
            linewidth=1.1,
            label="90% eligibility threshold"
        ),
    ]

    fig.suptitle(
        "Baseline training fit and seed stability",
        fontsize=13
    )

    fig.legend(
        handles=seed_handles,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(
            0.5,
            -0.01
        ),
        frameon=False
    )

    fig.tight_layout(
        rect=[
            0,
            0.10,
            1,
            0.95
        ]
    )

    save_figure(
        fig,
        "baseline_training_fit_by_seed"
    )

    plt.close(fig)


def save_eligible_baseline_plot(
    results,
    eligibility,
    task
):
    """
    Plot only baselines that learned the training task
    consistently across every seed.
    """

    eligible_rows = eligibility[
        (
            eligibility["Task"] == task
        )
        & (
            eligibility["Eligibility"]
            == "Eligible"
        )
    ].copy()

    if eligible_rows.empty:
        print(
            f"No eligible baselines for task: {task}"
        )

        return pd.DataFrame()

    eligible_encodings = [
        encoding
        for encoding in ENCODING_ORDER
        if encoding
        in eligible_rows[
            "Positional Encoding"
        ].unique()
    ]

    task_results = results[
        (
            results["Task"] == task
        )
        & (
            results["Positional Encoding"].isin(
                eligible_encodings
            )
        )
    ].copy()

    lengths = sorted(
        task_results["Test Length"].unique()
    )

    positions = length_positions(
        lengths
    )

    train_length = int(
        task_results["Train Length"].iloc[0]
    )

    train_position = positions[
        train_length
    ]

    fig, ax = plt.subplots(
        figsize=(7.4, 4.6)
    )

    style_axis(
        ax
    )

    ax.axvspan(
        train_position,
        len(lengths) - 1 + 0.42,
        color="#F3F5F7",
        alpha=0.70,
        zorder=0
    )

    add_training_length_marker(
        ax,
        train_position
    )

    add_collapse_threshold(
        ax
    )

    if len(eligible_encodings) == 1:
        marker_offsets = {
            eligible_encodings[0]: 0.0
        }
    else:
        offset_values = np.linspace(
            -0.06,
            0.06,
            len(eligible_encodings)
        )

        marker_offsets = {
            encoding: offset
            for encoding, offset
            in zip(
                eligible_encodings,
                offset_values
            )
        }

    legend_handles = []

    plot_data_frames = []

    for encoding in eligible_encodings:
        frame = task_results[
            task_results[
                "Positional Encoding"
            ]
            == encoding
        ].sort_values(
            [
                "Seed",
                "Test Length"
            ]
        )

        aggregate = aggregate_encoding(
            frame
        )

        aggregate[
            "Task"
        ] = task

        aggregate[
            "Task Label"
        ] = TASK_LABELS[task]

        aggregate[
            "Positional Encoding"
        ] = encoding

        aggregate[
            "Encoding Label"
        ] = ENCODING_LABELS[encoding]

        plot_data_frames.append(
            aggregate
        )

        true_x_values = [
            positions[int(length)]
            for length in aggregate["Test Length"]
        ]

        marker_x_values = [
            positions[int(length)]
            + marker_offsets[encoding]
            for length in aggregate["Test Length"]
        ]

        mean_values = aggregate[
            "Exact Mean"
        ].to_numpy()

        standard_deviation = aggregate[
            "Exact SD"
        ].to_numpy()

        lower_bound = np.maximum(
            mean_values - standard_deviation,
            0.0
        )

        upper_bound = np.minimum(
            mean_values + standard_deviation,
            100.0
        )

        # Faint curves show individual seeds.
        for seed in sorted(
            frame["Seed"].unique()
        ):
            seed_frame = frame[
                frame["Seed"] == seed
            ].sort_values(
                "Test Length"
            )

            seed_x_values = [
                positions[int(length)]
                for length
                in seed_frame["Test Length"]
            ]

            ax.plot(
                seed_x_values,
                seed_frame[
                    "Exact Match Accuracy"
                ],
                color=ENCODING_COLORS[
                    encoding
                ],
                linestyle=ENCODING_LINESTYLES[
                    encoding
                ],
                linewidth=0.9,
                alpha=0.20,
                zorder=2
            )

        # The shaded band represents ±1 standard deviation.
        ax.fill_between(
            true_x_values,
            lower_bound,
            upper_bound,
            color=ENCODING_COLORS[
                encoding
            ],
            alpha=0.12,
            linewidth=0,
            zorder=1
        )

        # Bold line shows the mean.
        ax.plot(
            true_x_values,
            mean_values,
            color=ENCODING_COLORS[
                encoding
            ],
            linestyle=ENCODING_LINESTYLES[
                encoding
            ],
            linewidth=2.2,
            alpha=0.98,
            zorder=3
        )

        # Mean markers and SD bars are slightly offset.
        ax.errorbar(
            marker_x_values,
            mean_values,
            yerr=standard_deviation,
            color=ENCODING_COLORS[
                encoding
            ],
            marker=ENCODING_MARKERS[
                encoding
            ],
            linestyle="none",
            markersize=5.5,
            markeredgecolor="white",
            markeredgewidth=0.65,
            capsize=2.8,
            linewidth=1.0,
            zorder=5
        )

        collapse_length = first_collapse_length(
            aggregate,
            train_length
        )

        if collapse_length is not None:
            collapse_row = aggregate[
                aggregate["Test Length"]
                == collapse_length
            ]

            if not collapse_row.empty:
                ax.scatter(
                    positions[collapse_length]
                    + marker_offsets[encoding],
                    collapse_row[
                        "Exact Mean"
                    ].iloc[0],
                    s=76,
                    facecolor="white",
                    edgecolor=ENCODING_COLORS[
                        encoding
                    ],
                    linewidth=1.5,
                    zorder=6
                )

        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=ENCODING_COLORS[
                    encoding
                ],
                marker=ENCODING_MARKERS[
                    encoding
                ],
                linestyle=ENCODING_LINESTYLES[
                    encoding
                ],
                linewidth=2.2,
                markersize=5.5,
                markeredgecolor="white",
                label=ENCODING_LABELS[
                    encoding
                ],
            )
        )

    ax.set_title(
        f"{TASK_LABELS[task]}: "
        "eligible baseline extrapolation"
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
        len(lengths) - 1 + 0.45
    )

    ax.set_xticks(
        range(len(lengths))
    )

    ax.set_xticklabels([
        str(length)
        for length in lengths
    ])

    ax.legend(
        handles=legend_handles,
        title=(
            "Eligible positional encoding\n"
            "Mean ± 1 SD; three seeds\n"
            "Faint curves: individual seeds"
        ),
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#C8CDD2"
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"{task}_eligible_baseline_extrapolation"
    )

    plt.close(fig)

    return pd.concat(
        plot_data_frames,
        ignore_index=True
    )


def main():
    """
    Generate corrected baseline diagnostic figures.
    """

    set_report_style()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = load_results()

    eligibility = build_baseline_eligibility(
        results
    )

    eligibility.to_csv(
        OUTPUT_DIR
        / "baseline_eligibility_summary.csv",
        index=False
    )

    save_training_fit_by_seed(
        results,
        eligibility
    )

    eligible_plot_data = []

    for task in TASK_ORDER:
        task_plot_data = (
            save_eligible_baseline_plot(
                results,
                eligibility,
                task
            )
        )

        if not task_plot_data.empty:
            eligible_plot_data.append(
                task_plot_data
            )

    if eligible_plot_data:
        combined_plot_data = pd.concat(
            eligible_plot_data,
            ignore_index=True
        )

        combined_plot_data.to_csv(
            OUTPUT_DIR
            / "eligible_baseline_plot_data.csv",
            index=False
        )

    print(
        "Saved baseline diagnostic PNGs, PDFs and CSVs "
        f"to: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()