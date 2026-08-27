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


# Restrained thesis palette.
# The same encoding always uses the same colour,
# marker and line style in every figure.
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

ENCODING_MARKER_OFFSETS = {
    "learned": -0.10,
    "sinusoidal": -0.05,
    "none": 0.00,
    "alibi": 0.05,
    "rope": 0.10,
}


# The mitigation colours belong to the same muted visual family.
# Marker and line-style differences preserve readability in grayscale.
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


# Every comparison points directly to the full raw result CSV.
# This prevents intermediate lengths from being discarded.
MITIGATION_COMPARISONS = [
    {
        "task": "addition",
        "encoding": "learned",
        "seed": 42,
        "output_stem": (
            "addition_learned_"
            "baseline_vs_mitigation"
        ),
        "target": (
            "Mixed-length training tests whether broader "
            "length exposure reduces length-specific learning."
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "path": (
                    MULTISEED_DIR
                    / "addition_train16_learned_seed42_results.csv"
                ),
            },
            {
                "label": "Mixed-length (v1)",
                "style": "mixed",
                "path": (
                    MITIGATION_DIR
                    / "addition_train16_mixed_learned_seed42_results.csv"
                ),
            },
            {
                "label": "Mixed-length (v2)",
                "style": "mixed_v2",
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
        "target": (
            "Mixed-length training tests length exposure; "
            "curriculum tests gradual optimisation; randomised "
            "padding tests dependence on absolute positions."
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "path": (
                    MULTISEED_DIR
                    / "copy_train128_sinusoidal_seed42_results.csv"
                ),
            },
            {
                "label": "Single-length control",
                "style": "control",
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_single_sinusoidal_seed42_control_results.csv"
                ),
            },
            {
                "label": "Mixed-length",
                "style": "mixed",
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_mixed_sinusoidal_seed42_results.csv"
                ),
            },
            {
                "label": "Curriculum",
                "style": "curriculum",
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_curriculum_sinusoidal_seed42_stage4_results.csv"
                ),
            },
            {
                "label": "Randomised padded",
                "style": "randomised",
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
        "target": (
            "Curriculum tests gradual optimisation; randomised "
            "padding tests dependence on fixed absolute positions."
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "path": (
                    MULTISEED_DIR
                    / "copy_train128_rope_seed42_results.csv"
                ),
            },
            {
                "label": "Curriculum",
                "style": "curriculum",
                "path": (
                    MITIGATION_DIR
                    / "copy_train128_curriculum_rope_seed42_stage4_results.csv"
                ),
            },
            {
                "label": "Randomised padded",
                "style": "randomised",
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
        "target": (
            "Curriculum training tests whether gradual increases "
            "in length improve optimisation and extrapolation."
        ),
        "series": [
            {
                "label": "Original baseline",
                "style": "baseline",
                "path": (
                    MULTISEED_DIR
                    / "reverse_train128_learned_seed42_results.csv"
                ),
            },
            {
                "label": "Curriculum",
                "style": "curriculum",
                "path": (
                    MITIGATION_DIR
                    / "reverse_train128_curriculum_learned_seed42_stage4_results.csv"
                ),
            },
        ],
    },
]


def set_report_style():
    """
    Apply one consistent thesis-wide visual style.
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


def style_line_axis(ax):
    """
    Apply consistent grid and axis styling to line plots.
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
    Save a high-resolution PNG and a vector PDF.

    PDF is preferred when inserting the plot into LaTeX.
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


def load_multiseed_results():
    """
    Load and combine all multi-seed baseline result files.
    """

    files = sorted(
        MULTISEED_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No multi-seed CSV files found in {MULTISEED_DIR}"
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
            "The multi-seed result files are missing columns: "
            f"{sorted(missing_columns)}"
        )

    results["Exact Match Error Rate"] = (
        100.0 - results["Exact Match Accuracy"]
    )

    results["Token Error Rate"] = (
        100.0 - results["Token Accuracy"]
    )

    return results


def aggregate_multiseed(results):
    """
    Calculate summary statistics across random seeds.
    """

    aggregate = results.groupby(
        [
            "Task",
            "Train Length",
            "Test Length",
            "Positional Encoding",
        ],
        as_index=False
    ).agg(
        Seeds=(
            "Seed",
            "nunique"
        ),
        Exact_Mean=(
            "Exact Match Accuracy",
            "mean"
        ),
        Exact_Std=(
            "Exact Match Accuracy",
            "std"
        ),
        Exact_SE=(
            "Exact Match Accuracy",
            "sem"
        ),
        Exact_Error_Mean=(
            "Exact Match Error Rate",
            "mean"
        ),
        Exact_Error_Std=(
            "Exact Match Error Rate",
            "std"
        ),
        Exact_Error_SE=(
            "Exact Match Error Rate",
            "sem"
        ),
        Token_Mean=(
            "Token Accuracy",
            "mean"
        ),
        Token_Std=(
            "Token Accuracy",
            "std"
        ),
        Token_SE=(
            "Token Accuracy",
            "sem"
        ),
        Token_Error_Mean=(
            "Token Error Rate",
            "mean"
        ),
        Token_Error_Std=(
            "Token Error Rate",
            "std"
        ),
        Token_Error_SE=(
            "Token Error Rate",
            "sem"
        ),
    )

    return aggregate.fillna(0.0)


def length_positions(lengths):
    """
    Map sequence lengths to equally spaced categorical positions.
    """

    return {
        length: index
        for index, length in enumerate(
            sorted(lengths)
        )
    }


def get_failure_length(
    frame,
    train_length,
    threshold=COLLAPSE_THRESHOLD
):
    """
    Return the first unseen length where mean exact-match
    accuracy falls below the collapse threshold.
    """

    extrapolation = frame[
        frame["Test Length"] > train_length
    ].sort_values(
        "Test Length"
    )

    failures = extrapolation[
        extrapolation["Exact_Mean"] < threshold
    ]

    if failures.empty:
        return None

    return int(
        failures["Test Length"].iloc[0]
    )


def add_train_marker(
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


def save_failure_behaviour_plot(
    aggregate,
    task
):
    """
    Plot exact-match accuracy across test lengths.
    """

    task_frame = aggregate[
        aggregate["Task"] == task
    ].copy()

    if task_frame.empty:
        return

    lengths = sorted(
        task_frame["Test Length"].unique()
    )

    positions = length_positions(
        lengths
    )

    train_length = int(
        task_frame["Train Length"].iloc[0]
    )

    train_position = positions[
        train_length
    ]

    fig, ax = plt.subplots(
        figsize=(7.4, 4.6)
    )

    style_line_axis(ax)

    ax.axvspan(
        train_position,
        len(lengths) - 1 + 0.42,
        color="#F3F5F7",
        alpha=0.70,
        zorder=0
    )

    add_train_marker(
        ax,
        train_position
    )

    add_collapse_threshold(
        ax
    )

    for encoding in ENCODING_ORDER:
        frame = task_frame[
            task_frame["Positional Encoding"]
            == encoding
        ].sort_values(
            "Test Length"
        )

        if frame.empty:
            continue

        x_values = [
            positions[length]
            + ENCODING_MARKER_OFFSETS[encoding]
            for length in frame["Test Length"]
        ]

        ax.errorbar(
            x_values,
            frame["Exact_Mean"],
            yerr=frame["Exact_SE"],
            label=ENCODING_LABELS[encoding],
            color=ENCODING_COLORS[encoding],
            marker=ENCODING_MARKERS[encoding],
            linestyle=ENCODING_LINESTYLES[encoding],
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            capsize=2.5,
            alpha=0.96,
            zorder=3,
        )

        failure_length = get_failure_length(
            frame,
            train_length
        )

        if failure_length is not None:
            failure_row = frame[
                frame["Test Length"]
                == failure_length
            ]

            if not failure_row.empty:
                ax.scatter(
                    positions[failure_length]
                    + ENCODING_MARKER_OFFSETS[encoding],
                    failure_row["Exact_Mean"].iloc[0],
                    s=64,
                    facecolor="white",
                    edgecolor=ENCODING_COLORS[encoding],
                    linewidth=1.4,
                    zorder=5
                )

    ax.set_title(
        f"{TASK_LABELS[task]}: "
        "exact-match accuracy across lengths"
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
        title="Positional encoding",
        loc="upper right",
        frameon=True,
        framealpha=0.96,
        edgecolor="#C8CDD2"
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"{task}_exact_failure_behaviour"
    )

    plt.close(fig)


def save_error_rate_heatmap(
    aggregate,
    task
):
    """
    Plot exact-match error percentages.

    A common red scale is used for every error heatmap.
    """

    task_frame = aggregate[
        aggregate["Task"] == task
    ].copy()

    if task_frame.empty:
        return

    lengths = sorted(
        task_frame["Test Length"].unique()
    )

    matrix = []

    available_encodings = []

    for encoding in ENCODING_ORDER:
        frame = task_frame[
            task_frame["Positional Encoding"]
            == encoding
        ]

        if frame.empty:
            continue

        available_encodings.append(
            encoding
        )

        row = []

        for length in lengths:
            value = frame[
                frame["Test Length"] == length
            ]["Exact_Error_Mean"]

            if value.empty:
                row.append(
                    np.nan
                )
            else:
                row.append(
                    value.iloc[0]
                )

        matrix.append(
            row
        )

    matrix = np.array(
        matrix
    )

    fig, ax = plt.subplots(
        figsize=(7.4, 3.8)
    )

    image = ax.imshow(
        matrix,
        cmap="Reds",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_title(
        f"{TASK_LABELS[task]}: "
        "exact-match error rate"
    )

    ax.set_xlabel(
        "Test sequence length"
    )

    ax.set_ylabel(
        "Positional encoding"
    )

    ax.set_xticks(
        range(len(lengths))
    )

    ax.set_xticklabels([
        str(length)
        for length in lengths
    ])

    ax.set_yticks(
        range(len(available_encodings))
    )

    ax.set_yticklabels([
        ENCODING_LABELS[encoding]
        for encoding in available_encodings
    ])

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index
            ]

            if np.isnan(value):
                display_value = "N/A"
                text_color = "#222222"
            else:
                display_value = f"{value:.1f}%"
                text_color = (
                    "white"
                    if value > 55
                    else "#222222"
                )

            ax.text(
                column_index,
                row_index,
                display_value,
                ha="center",
                va="center",
                fontsize=8.2,
                color=text_color
            )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Exact-match error rate (%)"
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"{task}_exact_error_rate_heatmap"
    )

    plt.close(fig)


def save_train_performance_heatmap(
    aggregate
):
    """
    Plot training-length exact-match accuracy.
    """

    matrix = []

    available_encodings = []

    for encoding in ENCODING_ORDER:
        row = []

        encoding_has_data = False

        for task in TASK_ORDER:
            frame = aggregate[
                (
                    aggregate["Task"]
                    == task
                )
                & (
                    aggregate["Positional Encoding"]
                    == encoding
                )
            ].sort_values(
                "Test Length"
            )

            if frame.empty:
                row.append(
                    np.nan
                )
                continue

            encoding_has_data = True

            train_length = int(
                frame["Train Length"].iloc[0]
            )

            train_row = frame[
                frame["Test Length"]
                == train_length
            ]

            if train_row.empty:
                row.append(
                    np.nan
                )
            else:
                row.append(
                    train_row["Exact_Mean"].iloc[0]
                )

        if encoding_has_data:
            available_encodings.append(
                encoding
            )

            matrix.append(
                row
            )

    matrix = np.array(
        matrix
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 4.2)
    )

    image = ax.imshow(
        matrix,
        cmap="Blues",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_title(
        "Training-length exact-match accuracy"
    )

    ax.set_xlabel(
        "Task"
    )

    ax.set_ylabel(
        "Positional encoding"
    )

    ax.set_xticks(
        range(len(TASK_ORDER))
    )

    ax.set_xticklabels([
        TASK_LABELS[task]
        for task in TASK_ORDER
    ])

    ax.set_yticks(
        range(len(available_encodings))
    )

    ax.set_yticklabels([
        ENCODING_LABELS[encoding]
        for encoding in available_encodings
    ])

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index
            ]

            if np.isnan(value):
                display_value = "N/A"
                text_color = "#222222"
            else:
                display_value = f"{value:.1f}%"
                text_color = (
                    "white"
                    if value > 55
                    else "#222222"
                )

            ax.text(
                column_index,
                row_index,
                display_value,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color
            )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Mean exact-match accuracy (%)"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "exact_at_train_heatmap"
    )

    plt.close(fig)


def save_generalisation_gap_heatmap(
    aggregate
):
    """
    Plot the exact-match generalisation gap.
    """

    matrix = []

    available_encodings = []

    for encoding in ENCODING_ORDER:
        row = []

        encoding_has_data = False

        for task in TASK_ORDER:
            frame = aggregate[
                (
                    aggregate["Task"]
                    == task
                )
                & (
                    aggregate["Positional Encoding"]
                    == encoding
                )
            ].sort_values(
                "Test Length"
            )

            if frame.empty:
                row.append(
                    np.nan
                )
                continue

            encoding_has_data = True

            train_length = int(
                frame["Train Length"].iloc[0]
            )

            train_row = frame[
                frame["Test Length"]
                == train_length
            ]

            if train_row.empty:
                row.append(
                    np.nan
                )
                continue

            longest_row = frame.iloc[-1]

            gap = (
                train_row["Exact_Mean"].iloc[0]
                - longest_row["Exact_Mean"]
            )

            row.append(
                gap
            )

        if encoding_has_data:
            available_encodings.append(
                encoding
            )

            matrix.append(
                row
            )

    matrix = np.array(
        matrix
    )

    fig, ax = plt.subplots(
        figsize=(7.0, 4.2)
    )

    image = ax.imshow(
        matrix,
        cmap="Reds",
        vmin=0,
        vmax=100,
        aspect="auto"
    )

    ax.set_title(
        "Generalisation gap: "
        "training length minus longest length"
    )

    ax.set_xlabel(
        "Task"
    )

    ax.set_ylabel(
        "Positional encoding"
    )

    ax.set_xticks(
        range(len(TASK_ORDER))
    )

    ax.set_xticklabels([
        TASK_LABELS[task]
        for task in TASK_ORDER
    ])

    ax.set_yticks(
        range(len(available_encodings))
    )

    ax.set_yticklabels([
        ENCODING_LABELS[encoding]
        for encoding in available_encodings
    ])

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            value = matrix[
                row_index,
                column_index
            ]

            if np.isnan(value):
                display_value = "N/A"
                text_color = "#222222"
            else:
                display_value = f"{value:.1f}"
                text_color = (
                    "white"
                    if value > 55
                    else "#222222"
                )

            ax.text(
                column_index,
                row_index,
                display_value,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color
            )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Exact-match percentage-point gap"
    )

    fig.tight_layout()

    save_figure(
        fig,
        "generalisation_gap_heatmap"
    )

    plt.close(fig)


def load_raw_experiment(
    path,
    expected_task,
    expected_encoding,
    expected_seed
):
    """
    Load and validate one complete raw experiment CSV.
    """

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

    frame["Token Error Rate"] = (
        100.0 - frame["Token Accuracy"]
    )

    frame["Exact Match Error Rate"] = (
        100.0 - frame["Exact Match Accuracy"]
    )

    return frame.sort_values(
        "Test Length"
    )


def save_baseline_vs_mitigation_plots():
    """
    Plot matched mitigation comparisons using complete raw CSVs.

    Lines remain at the true x positions. Only markers are
    horizontally offset to reveal coincident observations.
    """

    for comparison in MITIGATION_COMPARISONS:
        task = comparison["task"]

        encoding = comparison["encoding"]

        seed = comparison["seed"]

        plot_series = []

        train_lengths = set()

        plot_data_rows = []

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

            train_row = frame[
                frame["Test Length"]
                == train_length
            ]

            if train_row.empty:
                raise ValueError(
                    "Missing training-length evaluation in "
                    f"{series['path']}"
                )

            exact_at_train = float(
                train_row[
                    "Exact Match Accuracy"
                ].iloc[0]
            )

            if exact_at_train < TRAIN_FIT_THRESHOLD:
                legend_label = (
                    f"{series['label']} "
                    "(training underfit)"
                )

                training_status = (
                    "Training underfit"
                )
            else:
                legend_label = (
                    series["label"]
                )

                training_status = (
                    "Training task learned"
                )

            plot_series.append({
                "label": series["label"],
                "legend_label": legend_label,
                "style": series["style"],
                "frame": frame,
            })

            for _, row in frame.iterrows():
                plot_data_rows.append({
                    "Task": task,
                    "Task Label": TASK_LABELS[task],
                    "Positional Encoding": encoding,
                    "Positional Encoding Label": (
                        ENCODING_LABELS[encoding]
                    ),
                    "Training Condition": series["label"],
                    "Training Status": training_status,
                    "Seed": int(row["Seed"]),
                    "Train Length": int(row["Train Length"]),
                    "Test Length": int(row["Test Length"]),
                    "Token Accuracy (%)": row["Token Accuracy"],
                    "Token Error Rate (%)": row["Token Error Rate"],
                    "Exact-Match Accuracy (%)": (
                        row["Exact Match Accuracy"]
                    ),
                    "Exact-Match Error Rate (%)": (
                        row["Exact Match Error Rate"]
                    ),
                    "Mitigation Target": comparison["target"],
                    "Source CSV": str(series["path"]),
                })

        if len(train_lengths) != 1:
            raise ValueError(
                "Matched mitigation comparison contains "
                f"different training lengths: {train_lengths}"
            )

        train_length = next(
            iter(train_lengths)
        )

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

        add_train_marker(
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

            # Lines remain at the exact test-length positions.
            ax.plot(
                true_x_values,
                y_values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.1,
                alpha=0.95,
                zorder=2 + series_index,
            )

            # Only markers are offset for visibility.
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

        plt.close(fig)

        plot_data = pd.DataFrame(
            plot_data_rows
        )

        plot_data.to_csv(
            OUTPUT_DIR
            / (
                f"{comparison['output_stem']}"
                "_data.csv"
            ),
            index=False
        )


def save_failure_summary_table(
    results,
    aggregate
):
    """
    Create a baseline-validity and failure-behaviour table.
    """

    rows = []

    for task in TASK_ORDER:
        for encoding in ENCODING_ORDER:
            raw_frame = results[
                (
                    results["Task"]
                    == task
                )
                & (
                    results["Positional Encoding"]
                    == encoding
                )
            ].copy()

            aggregate_frame = aggregate[
                (
                    aggregate["Task"]
                    == task
                )
                & (
                    aggregate["Positional Encoding"]
                    == encoding
                )
            ].sort_values(
                "Test Length"
            )

            if (
                raw_frame.empty
                or aggregate_frame.empty
            ):
                continue

            train_length = int(
                raw_frame["Train Length"].iloc[0]
            )

            train_seed_rows = raw_frame[
                raw_frame["Test Length"]
                == train_length
            ].sort_values(
                "Seed"
            )

            if train_seed_rows.empty:
                continue

            longest_length = int(
                aggregate_frame["Test Length"].max()
            )

            longest_seed_rows = raw_frame[
                raw_frame["Test Length"]
                == longest_length
            ]

            seed_count = int(
                train_seed_rows["Seed"].nunique()
            )

            successful_seed_mask = (
                train_seed_rows[
                    "Exact Match Accuracy"
                ]
                >= TRAIN_FIT_THRESHOLD
            )

            successful_seed_count = int(
                train_seed_rows.loc[
                    successful_seed_mask,
                    "Seed"
                ].nunique()
            )

            successful_seed_values = (
                train_seed_rows.loc[
                    successful_seed_mask,
                    "Seed"
                ]
                .astype(str)
                .tolist()
            )

            train_mean = train_seed_rows[
                "Exact Match Accuracy"
            ].mean()

            train_std = train_seed_rows[
                "Exact Match Accuracy"
            ].std()

            if pd.isna(train_std):
                train_std = 0.0

            train_min = train_seed_rows[
                "Exact Match Accuracy"
            ].min()

            train_max = train_seed_rows[
                "Exact Match Accuracy"
            ].max()

            train_token_mean = train_seed_rows[
                "Token Accuracy"
            ].mean()

            longest_mean = longest_seed_rows[
                "Exact Match Accuracy"
            ].mean()

            longest_token_mean = longest_seed_rows[
                "Token Accuracy"
            ].mean()

            if successful_seed_count == seed_count:
                failure_length = get_failure_length(
                    aggregate_frame,
                    train_length,
                    threshold=COLLAPSE_THRESHOLD
                )

                eligibility = "Eligible"

                if failure_length is None:
                    behaviour = (
                        "Maintains extrapolation"
                    )

                    failure_display = (
                        "Not below threshold"
                    )
                else:
                    behaviour = (
                        "Extrapolation collapse"
                    )

                    failure_display = (
                        failure_length
                    )

            elif successful_seed_count == 0:
                failure_display = (
                    "Not applicable"
                )

                eligibility = (
                    "Not eligible"
                )

                if train_mean < COLLAPSE_THRESHOLD:
                    behaviour = (
                        "Training-length underfitting"
                    )
                else:
                    behaviour = (
                        "Partial training fit"
                    )

            else:
                failure_display = (
                    "Do not pool seeds"
                )

                behaviour = (
                    "Seed-instability at training length"
                )

                eligibility = (
                    "Analyse successful seeds separately"
                )

            successful_seeds_text = (
                ", ".join(
                    successful_seed_values
                )
                if successful_seed_values
                else "None"
            )

            rows.append({
                "Task": TASK_LABELS[task],
                "Encoding": ENCODING_LABELS[encoding],
                "Train Length": train_length,
                "Total Seeds": seed_count,
                "Train-fit Seeds": (
                    f"{successful_seed_count}/{seed_count}"
                ),
                "Successful Seed IDs": successful_seeds_text,
                "Exact@Train Mean (%)": round(
                    train_mean,
                    2
                ),
                "Exact@Train SD "
                "(percentage points)": round(
                    train_std,
                    2
                ),
                "Exact@Train Range (%)": (
                    f"{train_min:.2f}–{train_max:.2f}"
                ),
                "Token@Train Mean (%)": round(
                    train_token_mean,
                    2
                ),
                "Error@Train Mean (%)": round(
                    100.0 - train_mean,
                    2
                ),
                "Longest Test Length": longest_length,
                "Exact@Longest Mean (%)": round(
                    longest_mean,
                    2
                ),
                "Token@Longest Mean (%)": round(
                    longest_token_mean,
                    2
                ),
                "Error@Longest Mean (%)": round(
                    100.0 - longest_mean,
                    2
                ),
                "First Collapse Length": failure_display,
                "Failure Behaviour": behaviour,
                "Baseline Eligibility": eligibility,
            })

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTPUT_DIR
        / "failure_behaviour_summary.csv",
        index=False
    )


def main():
    """
    Generate all final report plots, vector figures,
    and plot-data CSV files.
    """

    set_report_style()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    multiseed = load_multiseed_results()

    aggregate = aggregate_multiseed(
        multiseed
    )

    for task in TASK_ORDER:
        save_failure_behaviour_plot(
            aggregate,
            task
        )

        save_error_rate_heatmap(
            aggregate,
            task
        )

    save_train_performance_heatmap(
        aggregate
    )

    save_generalisation_gap_heatmap(
        aggregate
    )

    save_failure_summary_table(
        multiseed,
        aggregate
    )

    save_baseline_vs_mitigation_plots()

    print(
        "Saved final PNG figures, PDF figures, "
        "summary tables and plot-data CSV files to: "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()