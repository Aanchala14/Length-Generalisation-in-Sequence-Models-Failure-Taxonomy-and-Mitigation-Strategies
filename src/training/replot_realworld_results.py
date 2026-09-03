from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "outputs/plots/realworld/"
    "selfregulationscp2_summary.csv"
)

OUTPUT_DIR = Path(
    "outputs/plots/final"
)

OUTPUT_STEM = (
    "selfregulationscp2_accuracy_mean_se"
)


ENCODING_ORDER = [
    "learned",
    "sinusoidal",
    "none",
    "alibi",
    "rope",
]

ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
    "none": "NoPE",
    "alibi": "ALiBi",
    "rope": "RoPE",
}


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

ENCODING_OFFSETS = {
    "learned": -8.0,
    "sinusoidal": -4.0,
    "none": 0.0,
    "alibi": 4.0,
    "rope": 8.0,
}


TRAIN_LENGTH = 256

UNIFORM_TWO_CLASS_REFERENCE = 50.0

EXPECTED_TEST_LENGTHS = [
    256,
    512,
    1024,
    1152,
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
        "legend.fontsize": 8.3,
        "legend.title_fontsize": 8.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(axis):

    axis.set_axisbelow(True)

    axis.grid(
        axis="y",
        color="#D7DCE2",
        linestyle="--",
        linewidth=0.7,
        alpha=0.70,
    )


def load_summary():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing real-world summary CSV: {INPUT_CSV}"
        )

    frame = pd.read_csv(
        INPUT_CSV
    )

    required_columns = {
        "Dataset",
        "Train Length",
        "Test Length",
        "Positional Encoding",
        "Seeds",
        "Mean Accuracy",
        "Std Accuracy",
        "SE Accuracy",
    }

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "The summary CSV is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    frame = frame.copy()

    frame["Positional Encoding"] = (
        frame["Positional Encoding"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    numeric_columns = [
        "Train Length",
        "Test Length",
        "Seeds",
        "Mean Accuracy",
        "Std Accuracy",
        "SE Accuracy",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise",
        )

    frame["Train Length"] = (
        frame["Train Length"].astype(int)
    )

    frame["Test Length"] = (
        frame["Test Length"].astype(int)
    )

    frame["Seeds"] = (
        frame["Seeds"].astype(int)
    )

    unexpected_encodings = (
        set(frame["Positional Encoding"])
        - set(ENCODING_ORDER)
    )

    if unexpected_encodings:
        raise ValueError(
            "Unexpected positional encodings in summary: "
            f"{sorted(unexpected_encodings)}"
        )

    observed_encodings = set(
        frame["Positional Encoding"]
    )

    missing_encodings = (
        set(ENCODING_ORDER)
        - observed_encodings
    )

    if missing_encodings:
        raise ValueError(
            "Missing positional encodings from summary: "
            f"{sorted(missing_encodings)}"
        )

    observed_lengths = sorted(
        frame["Test Length"].unique().tolist()
    )

    if observed_lengths != EXPECTED_TEST_LENGTHS:
        raise ValueError(
            "Unexpected evaluation lengths. Expected "
            f"{EXPECTED_TEST_LENGTHS}, found "
            f"{observed_lengths}."
        )

    if not (
        frame["Train Length"] == TRAIN_LENGTH
    ).all():
        raise ValueError(
            "The summary contains an unexpected training length."
        )

    if not (
        frame["Seeds"] == 3
    ).all():
        raise ValueError(
            "Every plotted estimate must contain three seeds."
        )

    duplicate_rows = frame.duplicated(
        subset=[
            "Positional Encoding",
            "Test Length",
        ],
        keep=False,
    )

    if duplicate_rows.any():
        duplicate_values = frame.loc[
            duplicate_rows,
            [
                "Positional Encoding",
                "Test Length",
            ],
        ]

        raise ValueError(
            "Duplicate encoding-length rows found:\n"
            f"{duplicate_values.to_string(index=False)}"
        )

    expected_pairs = {
        (encoding, test_length)
        for encoding in ENCODING_ORDER
        for test_length in EXPECTED_TEST_LENGTHS
    }

    observed_pairs = set(
        zip(
            frame["Positional Encoding"],
            frame["Test Length"],
        )
    )

    missing_pairs = expected_pairs - observed_pairs

    if missing_pairs:
        raise ValueError(
            "Missing encoding-length combinations: "
            f"{sorted(missing_pairs)}"
        )

    return frame


def add_reference_lines(axis):


    axis.axvspan(
        TRAIN_LENGTH,
        1200,
        color="#F2F4F7",
        alpha=0.70,
        zorder=0,
    )

    axis.axvline(
        TRAIN_LENGTH,
        color="#4D4D4D",
        linestyle="--",
        linewidth=1.4,
        zorder=1,
    )

    axis.axhline(
        UNIFORM_TWO_CLASS_REFERENCE,
        color="#555555",
        linestyle=":",
        linewidth=1.8,
        zorder=2,
    )

    axis.annotate(
        "training prefix length",
        xy=(TRAIN_LENGTH, 58.1),
        xytext=(12, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=8.5,
        color="#4D4D4D",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#C8CDD2",
            "alpha": 0.95,
        },
    )

    axis.text(
        0.985,
        0.025,
        "Dotted line: uniform two-class reference (50%)",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.2,
        color="#555555",
        bbox={
            "boxstyle": "round,pad=0.18",
            "facecolor": "white",
            "edgecolor": "#C8CDD2",
            "alpha": 0.95,
        },
        zorder=5,
    )


def plot_results(frame):

    set_report_style()

    fig, axis = plt.subplots(
        figsize=(9.4, 5.7)
    )

    style_axis(
        axis
    )

    add_reference_lines(
        axis
    )

    for encoding in ENCODING_ORDER:
        encoding_frame = (
            frame[
                frame["Positional Encoding"]
                == encoding
            ]
            .sort_values("Test Length")
        )

        nominal_x = encoding_frame[
            "Test Length"
        ].to_numpy(dtype=float)

        plotted_x = (
            nominal_x
            + ENCODING_OFFSETS[encoding]
        )

        mean_accuracy = encoding_frame[
            "Mean Accuracy"
        ].to_numpy(dtype=float)

        standard_error = encoding_frame[
            "SE Accuracy"
        ].to_numpy(dtype=float)

        axis.errorbar(
            plotted_x,
            mean_accuracy,
            yerr=standard_error,
            color=ENCODING_COLORS[encoding],
            marker=ENCODING_MARKERS[encoding],
            linestyle=ENCODING_LINESTYLES[encoding],
            linewidth=2.0,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            capsize=3.5,
            capthick=1.2,
            elinewidth=1.2,
            label=ENCODING_LABELS[encoding],
            zorder=3,
        )

    axis.set_title(
        "SelfRegulationSCP2: accuracy across input lengths",
        pad=11,
    )

    axis.set_xlabel(
        "Evaluation prefix length"
    )

    axis.set_ylabel(
        "Classification accuracy (%)"
    )

    axis.set_xticks(
        EXPECTED_TEST_LENGTHS
    )

    axis.set_xticklabels(
        [str(length) for length in EXPECTED_TEST_LENGTHS]
    )

    axis.set_xlim(
        210,
        1200,
    )


    axis.set_ylim(
        45,
        59,
    )

    axis.set_yticks(
        np.arange(46, 60, 2)
    )

    encoding_handles = [
        Line2D(
            [0],
            [0],
            color=ENCODING_COLORS[encoding],
            marker=ENCODING_MARKERS[encoding],
            linestyle=ENCODING_LINESTYLES[encoding],
            linewidth=2.0,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=ENCODING_LABELS[encoding],
        )
        for encoding in ENCODING_ORDER
    ]

    legend = axis.legend(
        handles=encoding_handles,
        title=(
            "Positional encoding "
            "(mean ± 1 SE; three seeds)"
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.19),
        ncol=5,
        frameon=True,
        fancybox=False,
        edgecolor="#C8CDD2",
        columnspacing=1.5,
        handlelength=2.7,
    )

    legend.get_frame().set_facecolor(
        "white"
    )

    legend.get_frame().set_alpha(
        0.97
    )

    fig.text(
        0.5,
        0.015,
        (
            "Markers are offset slightly horizontally "
            "to reveal coincident estimates; "
            "ticks show nominal prefix lengths."
        ),
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#5A5A5A",
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.90,
        bottom=0.28,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        OUTPUT_DIR
        / f"{OUTPUT_STEM}.png"
    )

    pdf_path = (
        OUTPUT_DIR
        / f"{OUTPUT_STEM}.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    return png_path, pdf_path


def main():
    frame = load_summary()

    png_path, pdf_path = plot_results(
        frame
    )

    print(
        "Existing experimental values loaded from:"
    )
    print(
        f"  {INPUT_CSV}"
    )

    print(
        "Saved polished real-world figures:"
    )
    print(
        f"  {png_path}"
    )
    print(
        f"  {pdf_path}"
    )


if __name__ == "__main__":
    main()
