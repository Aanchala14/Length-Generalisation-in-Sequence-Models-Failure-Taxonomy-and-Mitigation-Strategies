from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "position_diagnostics_aws/final/"
    "addition_carry_condition_error.csv"
)

OUTPUT_DIR = Path(
    "position_diagnostics_aws/final"
)


ENCODING_ORDER = [
    "learned",
    "sinusoidal",
]


ENCODING_LABELS = {
    "learned": "Learned",
    "sinusoidal": "Sinusoidal",
}


CONDITION_STYLES = {
    "No carry-in": {
        "label": "No incoming carry",
        "color": "#1F4E79",
        "marker": "o",
        "linestyle": "-",
        "offset": -0.04,
    },
    "Carry-in": {
        "label": "Incoming carry",
        "color": "#2E6F75",
        "marker": "s",
        "linestyle": "--",
        "offset": 0.04,
    },
}


UNIFORM_DIGIT_ERROR_REFERENCE = 90.0


def set_report_style():

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
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


def style_axis(axis):

    axis.set_axisbelow(True)

    axis.grid(
        axis="y",
        color="#D7DCE2",
        linestyle="--",
        linewidth=0.7,
        alpha=0.70
    )


def load_carry_data():

    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing carry diagnostic CSV: {INPUT_CSV}"
        )

    frame = pd.read_csv(
        INPUT_CSV
    )

    required_columns = {
        "Task",
        "Positional Encoding",
        "Encoding Label",
        "Seed",
        "Train Length",
        "Test Length",
        "Carry Condition",
        "Token Count",
        "Error Count",
        "Error Rate (%)",
    }

    missing_columns = (
        required_columns - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Carry diagnostic CSV is missing columns: "
            f"{sorted(missing_columns)}"
        )

    tasks = set(
        frame["Task"].unique()
    )

    if tasks != {"addition"}:
        raise ValueError(
            "Expected only the addition task, "
            f"but found: {tasks}"
        )

    numeric_columns = [
        "Seed",
        "Train Length",
        "Test Length",
        "Token Count",
        "Error Count",
        "Error Rate (%)",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="raise"
        )

    return frame


def create_comparison_summary(
    carry_data
):

    ordinary_digits = carry_data[
        carry_data["Carry Condition"].isin(
            CONDITION_STYLES
        )
    ].copy()

    summary = ordinary_digits.pivot_table(
        index=[
            "Task",
            "Positional Encoding",
            "Encoding Label",
            "Seed",
            "Train Length",
            "Test Length",
        ],
        columns="Carry Condition",
        values="Error Rate (%)",
        aggfunc="first"
    ).reset_index()

    required_conditions = {
        "No carry-in",
        "Carry-in",
    }

    missing_conditions = (
        required_conditions - set(summary.columns)
    )

    if missing_conditions:
        raise ValueError(
            "Cannot compare carry conditions because these "
            f"columns are missing: {sorted(missing_conditions)}"
        )

    summary["Carry Minus No-Carry (percentage points)"] = (
        summary["Carry-in"]
        - summary["No carry-in"]
    )

    summary[
        "Absolute Carry Difference (percentage points)"
    ] = (
        summary[
            "Carry Minus No-Carry (percentage points)"
        ].abs()
    )

    summary["Interpretation"] = np.where(
        summary[
            "Absolute Carry Difference (percentage points)"
        ] < 1.0,
        (
            "Carry and no-carry error differ by less than "
            "one percentage point"
        ),
        (
            "Carry and no-carry error differ by at least "
            "one percentage point"
        )
    )

    summary = summary.rename(
        columns={
            "No carry-in": (
                "No Incoming Carry Error (%)"
            ),
            "Carry-in": (
                "Incoming Carry Error (%)"
            ),
        }
    )

    summary = summary.sort_values(
        [
            "Positional Encoding",
            "Test Length",
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / "addition_carry_comparison_summary.csv",
        index=False
    )

    return summary


def add_training_marker(
    axis,
    train_position
):

    axis.axvline(
        train_position,
        color="#3F3F3F",
        linestyle="--",
        linewidth=1.1,
        zorder=1
    )

    axis.annotate(
        "training length",
        xy=(
            train_position,
            102
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
        zorder=7
    )


def add_uniform_reference(
    axis
):

    axis.axhline(
        UNIFORM_DIGIT_ERROR_REFERENCE,
        color="#777777",
        linestyle=":",
        linewidth=1.1,
        alpha=0.90,
        zorder=1
    )

    axis.annotate(
        "uniform digit reference (90% error)",
        xy=(
            0.98,
            UNIFORM_DIGIT_ERROR_REFERENCE
        ),
        xycoords=(
            "axes fraction",
            "data"
        ),
        xytext=(
            0,
            -11
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
        zorder=7
    )


def save_carry_comparison_plot(
    carry_data
):

    ordinary_digits = carry_data[
        carry_data["Carry Condition"].isin(
            CONDITION_STYLES
        )
    ].copy()

    encodings = [
        encoding
        for encoding in ENCODING_ORDER
        if encoding in set(
            ordinary_digits[
                "Positional Encoding"
            ]
        )
    ]

    if not encodings:
        raise ValueError(
            "No supported positional encodings found."
        )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(encodings),
        figsize=(
            5.5 * len(encodings),
            4.7
        ),
        sharey=True,
        squeeze=False,
        constrained_layout=True
    )

    axes = axes.ravel()

    for axis, encoding in zip(
        axes,
        encodings
    ):
        encoding_frame = ordinary_digits[
            ordinary_digits[
                "Positional Encoding"
            ] == encoding
        ].copy()

        train_lengths = set(
            encoding_frame[
                "Train Length"
            ].astype(int)
        )

        if len(train_lengths) != 1:
            raise ValueError(
                f"{encoding} has inconsistent training lengths: "
                f"{train_lengths}"
            )

        train_length = next(
            iter(train_lengths)
        )

        lengths = sorted(
            encoding_frame[
                "Test Length"
            ].astype(int).unique()
        )

        positions = {
            length: index
            for index, length in enumerate(
                lengths
            )
        }

        train_position = positions[
            train_length
        ]

        axis.axvspan(
            train_position + 0.5,
            len(lengths) - 1 + 0.45,
            color="#F3F5F7",
            alpha=0.70,
            zorder=0
        )

        add_training_marker(
            axis,
            train_position
        )

        add_uniform_reference(
            axis
        )

        for condition, style in (
            CONDITION_STYLES.items()
        ):
            condition_frame = (
                encoding_frame[
                    encoding_frame[
                        "Carry Condition"
                    ] == condition
                ]
                .set_index(
                    "Test Length"
                )
                .reindex(
                    lengths
                )
            )

            if condition_frame[
                "Error Rate (%)"
            ].isna().any():
                raise ValueError(
                    f"{encoding} is missing values for "
                    f"condition: {condition}"
                )

            true_x_values = [
                positions[length]
                for length in lengths
            ]

            marker_x_values = [
                positions[length]
                + style["offset"]
                for length in lengths
            ]

            error_values = condition_frame[
                "Error Rate (%)"
            ].to_numpy()

            axis.plot(
                true_x_values,
                error_values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                alpha=0.98,
                zorder=3
            )

            axis.scatter(
                marker_x_values,
                error_values,
                color=style["color"],
                marker=style["marker"],
                s=44,
                edgecolors="white",
                linewidths=0.65,
                alpha=0.98,
                zorder=5
            )

        axis.set_title(
            ENCODING_LABELS[
                encoding
            ]
        )

        axis.set_xlabel(
            "Test digit length"
        )

        axis.set_xlim(
            -0.45,
            len(lengths) - 1 + 0.45
        )

        axis.set_ylim(
            -3,
            106
        )

        axis.set_xticks(
            range(len(lengths))
        )

        axis.set_xticklabels([
            str(length)
            for length in lengths
        ])

        style_axis(
            axis
        )

    axes[0].set_ylabel(
        "Ordinary-digit error rate (%)"
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=style["label"],
        )
        for style in CONDITION_STYLES.values()
    ]

    figure.legend(
        handles=legend_handles,
        title="Target digit condition",
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.045
        ),
        ncol=2,
        frameon=False
    )

    figure.suptitle(
        (
            "Addition: ordinary-digit error with and "
            "without incoming carry "
            "(representative seed 42)"
        ),
        fontsize=12.5
    )

    output_stem = (
        "addition_carry_condition_error"
    )

    figure.savefig(
        OUTPUT_DIR
        / f"{output_stem}.png",
        bbox_inches="tight",
        facecolor="white"
    )

    figure.savefig(
        OUTPUT_DIR
        / f"{output_stem}.pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(
        figure
    )


def main():

    set_report_style()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    carry_data = load_carry_data()

    summary = create_comparison_summary(
        carry_data
    )

    save_carry_comparison_plot(
        carry_data
    )

    first_unseen_rows = summary[
        summary["Test Length"]
        > summary["Train Length"]
    ].sort_values(
        [
            "Positional Encoding",
            "Test Length",
        ]
    ).groupby(
        "Positional Encoding",
        as_index=False
    ).first()

    print(
        "Saved corrected carry plot and summary to: "
        f"{OUTPUT_DIR}"
    )

    print(
        "\nFirst-unseen carry comparison:"
    )

    print(
        first_unseen_rows[[
            "Encoding Label",
            "Test Length",
            "No Incoming Carry Error (%)",
            "Incoming Carry Error (%)",
            (
                "Carry Minus No-Carry "
                "(percentage points)"
            ),
        ]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()