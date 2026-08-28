from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUTPUT_DIR = Path("outputs/plots/final")
OUTPUT_STEM = "controlled_task_examples"


# Same restrained palette used in the final thesis figures.
TEXT_COLOR = "#222222"
MUTED_TEXT = "#58616A"

INPUT_EDGE = "#1F4E79"
INPUT_FILL = "#DCE8F2"

TARGET_EDGE = "#2E6F75"
TARGET_FILL = "#DDEBEC"

SPECIAL_EDGE = "#707070"
SPECIAL_FILL = "#ECEFF1"

PANEL_FILL = "#F7F9FB"
PANEL_EDGE = "#D7DCE2"
ARROW_COLOR = "#4A4A4A"


def set_report_style():
    """
    Apply the same typography used by the final thesis figures.
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
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def token_style(token, is_target):
    """
    Return coherent colours for a token cell.
    """

    if token in {"PAD", "SEP", "+"}:
        return SPECIAL_FILL, SPECIAL_EDGE

    if is_target:
        return TARGET_FILL, TARGET_EDGE

    return INPUT_FILL, INPUT_EDGE


def draw_token(
    ax,
    x,
    y,
    token,
    is_target=False,
    width=0.043,
    height=0.100,
):
    """
    Draw one labelled token cell.
    """

    facecolor, edgecolor = token_style(
        token=token,
        is_target=is_target,
    )

    cell = Rectangle(
        (x, y - height / 2),
        width,
        height,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.15,
        zorder=3,
    )

    ax.add_patch(cell)

    font_size = 7.5 if token in {"PAD", "SEP"} else 9.2

    ax.text(
        x + width / 2,
        y,
        token,
        ha="center",
        va="center",
        fontsize=font_size,
        color=TEXT_COLOR,
        zorder=4,
    )


def draw_sequence(
    ax,
    centre_x,
    y,
    tokens,
    is_target=False,
    width=0.043,
    gap=0.005,
):
    """
    Draw a centred sequence and return its horizontal boundaries.
    """

    total_width = (
        len(tokens) * width
        + (len(tokens) - 1) * gap
    )

    start_x = centre_x - total_width / 2
    end_x = start_x + total_width

    for index, token in enumerate(tokens):
        token_x = start_x + index * (width + gap)

        draw_token(
            ax=ax,
            x=token_x,
            y=y,
            token=token,
            is_target=is_target,
            width=width,
        )

    return start_x, end_x


def draw_panel(
    ax,
    y,
    task_name,
    task_description,
    input_tokens,
    target_tokens,
):
    """
    Draw one complete task row.
    """

    panel_height = 0.225
    panel_bottom = y - panel_height / 2

    panel = FancyBboxPatch(
        (0.015, panel_bottom),
        0.970,
        panel_height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        facecolor=PANEL_FILL,
        edgecolor=PANEL_EDGE,
        linewidth=0.9,
        zorder=0,
    )

    ax.add_patch(panel)

    # Dedicated task-label column.
    ax.text(
        0.045,
        y + 0.038,
        task_name,
        ha="left",
        va="center",
        fontsize=10.5,
        fontweight="semibold",
        color=TEXT_COLOR,
    )

    ax.text(
        0.045,
        y - 0.035,
        task_description,
        ha="left",
        va="center",
        fontsize=8.3,
        linespacing=1.25,
        color=MUTED_TEXT,
    )

    input_left, input_right = draw_sequence(
        ax=ax,
        centre_x=0.405,
        y=y,
        tokens=input_tokens,
        is_target=False,
    )

    target_left, target_right = draw_sequence(
        ax=ax,
        centre_x=0.805,
        y=y,
        tokens=target_tokens,
        is_target=True,
    )

    # Calculate the arrow from the sequence boundaries.
    # This prevents overlap for sequences of different lengths.
    arrow_start = input_right + 0.015
    arrow_end = target_left - 0.015

    ax.annotate(
        "",
        xy=(arrow_end, y),
        xytext=(arrow_start, y),
        arrowprops={
            "arrowstyle": "-|>",
            "color": ARROW_COLOR,
            "linewidth": 1.4,
            "mutation_scale": 12,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=4,
    )


def add_legend(ax):
    """
    Add a compact visual key.
    """

    legend_y = 0.025
    width = 0.025
    height = 0.037

    legend_items = [
        (
            0.270,
            INPUT_FILL,
            INPUT_EDGE,
            "Input content token",
        ),
        (
            0.500,
            TARGET_FILL,
            TARGET_EDGE,
            "Target content token",
        ),
        (
            0.730,
            SPECIAL_FILL,
            SPECIAL_EDGE,
            "Special token",
        ),
    ]

    for x, facecolor, edgecolor, label in legend_items:
        patch = Rectangle(
            (x, legend_y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.0,
        )

        ax.add_patch(patch)

        ax.text(
            x + width + 0.010,
            legend_y + height / 2,
            label,
            ha="left",
            va="center",
            fontsize=8.0,
            color=MUTED_TEXT,
        )


def create_figure():
    """
    Create the controlled-task schematic.
    """

    set_report_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(
        figsize=(9.2, 4.6)
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    ax.text(
        0.405,
        0.960,
        "Input sequence",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="semibold",
        color=TEXT_COLOR,
    )

    ax.text(
        0.805,
        0.960,
        "Target sequence",
        ha="center",
        va="center",
        fontsize=10.2,
        fontweight="semibold",
        color=TEXT_COLOR,
    )

    draw_panel(
        ax=ax,
        y=0.795,
        task_name="Addition",
        task_description=(
            "digit-wise computation\n"
            "and carry propagation"
        ),
        input_tokens=["3", "7", "+", "2", "8"],
        target_tokens=["PAD", "PAD", "PAD", "6", "5"],
    )

    draw_panel(
        ax=ax,
        y=0.510,
        task_name="Delayed Copy",
        task_description=(
            "retention and aligned\n"
            "retrieval after a delay"
        ),
        input_tokens=[
            "4",
            "1",
            "7",
            "SEP",
            "PAD",
            "PAD",
            "PAD",
        ],
        target_tokens=[
            "PAD",
            "PAD",
            "PAD",
            "PAD",
            "4",
            "1",
            "7",
        ],
    )

    draw_panel(
        ax=ax,
        y=0.225,
        task_name="Reverse",
        task_description=(
            "length-dependent\n"
            "positional transformation"
        ),
        input_tokens=["4", "1", "7", "3"],
        target_tokens=["3", "7", "1", "4"],
    )

    add_legend(ax)

    png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"

    fig.savefig(
        png_path,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.08,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
        facecolor="white",
        pad_inches=0.08,
    )

    plt.close(fig)

    print(f"Saved PNG preview to: {png_path}")
    print(f"Saved vector PDF to: {pdf_path}")


if __name__ == "__main__":
    create_figure()