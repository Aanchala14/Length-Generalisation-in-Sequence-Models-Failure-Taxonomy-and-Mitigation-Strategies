from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(
    "outputs/results/multiseed_results"
)

ELIGIBILITY_PATH = Path(
    "outputs/plots/final/baseline_eligibility_summary.csv"
)

OUTPUT_DIR = Path(
    "outputs/plots/final"
)


COLLAPSE_THRESHOLD = 10.0

SEQUENCE_ERROR_COLLAPSE_THRESHOLD = (
    100.0 - COLLAPSE_THRESHOLD
)


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


UNIFORM_TOKEN_ERROR_REFERENCE = {
    "addition": 90.0,
    "copy": 99.0,
    "reverse": 99.0,
}


def set_report_style():


    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11.5,
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


def load_results():

    files = sorted(
        RESULTS_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No result files found in {RESULTS_DIR}"
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

    results["Sequence Error Rate"] = (
        100.0 - results["Exact Match Accuracy"]
    )

    results["Token Error Rate"] = (
        100.0 - results["Token Accuracy"]
    )

    return results


def load_eligibility():

    if not ELIGIBILITY_PATH.exists():
        raise FileNotFoundError(
            "Run plot_baseline_diagnostics first. "
            f"Missing: {ELIGIBILITY_PATH}"
        )

    eligibility = pd.read_csv(
        ELIGIBILITY_PATH
    )

    required_columns = {
        "Task",
        "Positional Encoding",
        "Eligibility",
    }

    missing_columns = (
        required_columns - set(eligibility.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing eligibility columns: "
            f"{sorted(missing_columns)}"
        )

    return eligibility


def length_positions(lengths):

    return {
        length: index
        for index, length in enumerate(
            sorted(lengths)
        )
    }


def aggregate_metric(
    frame,
    metric
):

    aggregate = frame.groupby(
        "Test Length",
        as_index=False
    ).agg(
        Seed_Count=(
            "Seed",
            "nunique"
        ),
        Mean=(
            metric,
            "mean"
        ),
        SD=(
            metric,
            "std"
        ),
        Minimum=(
            metric,
            "min"
        ),
        Maximum=(
            metric,
            "max"
        ),
    )

    aggregate["SD"] = (
        aggregate["SD"].fillna(0.0)
    )

    return aggregate.sort_values(
        "Test Length"
    )


def get_eligible_encodings(
    eligibility,
    task
):

    task_eligibility = eligibility[
        (
            eligibility["Task"] == task
        )
        & (
            eligibility["Eligibility"]
            == "Eligible"
        )
    ]

    available = set(
        task_eligibility[
            "Positional Encoding"
        ].tolist()
    )

    return [
        encoding
        for encoding in ENCODING_ORDER
        if encoding in available
    ]


def add_training_marker(
    ax,
    train_position,
    show_label
):
    ax.axvline(
        train_position,
        color="#3F3F3F",
        linestyle="--",
        linewidth=1.1,
        zorder=1
    )

    if show_label:
        ax.annotate(
            "training length",
            xy=(
                train_position,
                103
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


def add_sequence_error_threshold(ax):

    ax.axhline(
        SEQUENCE_ERROR_COLLAPSE_THRESHOLD,
        color="#666666",
        linestyle=":",
        linewidth=1.1,
        alpha=0.90,
        zorder=1
    )

    ax.annotate(
        "collapse threshold",
        xy=(
            0.98,
            0.84
        ),
        xycoords="axes fraction",
        ha="right",
        va="center",
        fontsize=8.0,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.90,
        },
    )


def add_uniform_token_reference(
    ax,
    task
):

    reference = (
        UNIFORM_TOKEN_ERROR_REFERENCE[task]
    )

    ax.axhline(
        reference,
        color="#777777",
        linestyle=":",
        linewidth=1.1,
        alpha=0.90,
        zorder=1
    )

    ax.annotate(
        (
            "uniform valid-token reference "
            f"({reference:.0f}% error)"
        ),
        xy=(
            0.98,
            reference
        ),
        xycoords=(
            "axes fraction",
            "data"
        ),
        xytext=(
            0,
            -12
        ),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=8.0,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.94,
        },
        zorder=6
    )


def plot_metric(
    ax,
    results,
    task,
    eligible_encodings,
    positions,
    metric,
    marker_offsets,
    output_rows
):

    for encoding in eligible_encodings:
        frame = results[
            (
                results["Task"] == task
            )
            & (
                results["Positional Encoding"]
                == encoding
            )
        ].sort_values(
            [
                "Seed",
                "Test Length"
            ]
        )

        aggregate = aggregate_metric(
            frame,
            metric
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
            "Mean"
        ].to_numpy()

        standard_deviation = aggregate[
            "SD"
        ].to_numpy()

        lower_bound = np.maximum(
            mean_values - standard_deviation,
            0.0
        )

        upper_bound = np.minimum(
            mean_values + standard_deviation,
            100.0
        )


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
                seed_frame[metric],
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

        ax.scatter(
            marker_x_values,
            mean_values,
            color=ENCODING_COLORS[
                encoding
            ],
            marker=ENCODING_MARKERS[
                encoding
            ],
            s=42,
            edgecolors="white",
            linewidths=0.65,
            zorder=5
        )

        for _, row in aggregate.iterrows():
            output_rows.append({
                "Task": task,
                "Task Label": TASK_LABELS[task],
                "Positional Encoding": encoding,
                "Encoding Label": (
                    ENCODING_LABELS[encoding]
                ),
                "Metric": metric,
                "Test Length": int(
                    row["Test Length"]
                ),
                "Seed Count": int(
                    row["Seed_Count"]
                ),
                "Mean Error Rate (%)": round(
                    row["Mean"],
                    4
                ),
                "SD (percentage points)": round(
                    row["SD"],
                    4
                ),
                "Minimum Error Rate (%)": round(
                    row["Minimum"],
                    4
                ),
                "Maximum Error Rate (%)": round(
                    row["Maximum"],
                    4
                ),
            })


def save_task_error_plot(
    results,
    eligibility,
    task
):

    eligible_encodings = (
        get_eligible_encodings(
            eligibility,
            task
        )
    )

    if not eligible_encodings:
        print(
            f"No eligible encodings for task: {task}"
        )

        return pd.DataFrame()

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

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(12.0, 4.8),
        sharey=True
    )

    output_rows = []

    metric_settings = [
        {
            "metric": "Sequence Error Rate",
            "title": (
                "Sequence error "
                "(100 − exact-match accuracy)"
            ),
        },
        {
            "metric": "Token Error Rate",
            "title": (
                "Token error "
                "(100 − token accuracy)"
            ),
        },
    ]

    for axis_index, settings in enumerate(
        metric_settings
    ):
        ax = axes[
            axis_index
        ]

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

        add_training_marker(
            ax,
            train_position,
            show_label=(
                axis_index == 0
            )
        )

        if (
            settings["metric"]
            == "Sequence Error Rate"
        ):
            add_sequence_error_threshold(
                ax
            )
        else:
            add_uniform_token_reference(
                ax,
                task
            )

        plot_metric(
            ax=ax,
            results=results,
            task=task,
            eligible_encodings=eligible_encodings,
            positions=positions,
            metric=settings["metric"],
            marker_offsets=marker_offsets,
            output_rows=output_rows
        )

        ax.set_title(
            settings["title"]
        )

        ax.set_xlabel(
            "Test sequence length"
        )

        ax.set_ylim(
            -3,
            106
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

    axes[0].set_ylabel(
        "Error rate (%)"
    )

    legend_handles = [
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
        for encoding in eligible_encodings
    ]

    fig.suptitle(
        f"{TASK_LABELS[task]}: "
        "sequence-level and token-level error",
        fontsize=13
    )

    fig.legend(
        handles=legend_handles,
        title=(
            "Eligible positional encoding\n"
            "Mean ± 1 SD; three seeds\n"
            "Faint curves: individual seeds"
        ),
        loc="lower center",
        ncol=max(
            1,
            len(eligible_encodings)
        ),
        bbox_to_anchor=(
            0.5,
            -0.02
        ),
        frameon=False
    )

    fig.tight_layout(
        rect=[
            0,
            0.14,
            1,
            0.94
        ]
    )

    output_stem = (
        f"{task}_eligible_baseline_"
        "error_decomposition"
    )

    save_figure(
        fig,
        output_stem
    )

    plt.close(fig)

    return pd.DataFrame(
        output_rows
    )


def main():

    set_report_style()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results = load_results()

    eligibility = load_eligibility()

    output_frames = []

    for task in TASK_ORDER:
        task_output = save_task_error_plot(
            results,
            eligibility,
            task
        )

        if not task_output.empty:
            output_frames.append(
                task_output
            )

    if output_frames:
        combined_output = pd.concat(
            output_frames,
            ignore_index=True
        )

        combined_output.to_csv(
            OUTPUT_DIR
            / "eligible_baseline_error_decomposition.csv",
            index=False
        )

    print(
        "Saved error-decomposition PNGs, PDFs and CSV to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()