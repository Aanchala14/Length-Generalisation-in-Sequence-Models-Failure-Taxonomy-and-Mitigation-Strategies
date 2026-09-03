from collections import defaultdict
from pathlib import Path

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataloader import get_dataloader
from src.models.transformer import TransformerModel
from src.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "plots"
    / "final"
)

RERUN_RESULTS_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "position_diagnostic_reruns"
)

NUMBER_OF_POSITION_BINS = 10

SEED = 42


EXPERIMENTS = [
    {
        "config": (
            PROJECT_ROOT
            / "configs"
            / "multiseed"
            / "addition_learned_seed42.yaml"
        ),
    },
    {
        "config": (
            PROJECT_ROOT
            / "configs"
            / "multiseed"
            / "addition_sinusoidal_seed42.yaml"
        ),
    },
    {
        "config": (
            PROJECT_ROOT
            / "configs"
            / "multiseed"
            / "copy_sinusoidal_seed42.yaml"
        ),
    },
    {
        "config": (
            PROJECT_ROOT
            / "configs"
            / "multiseed"
            / "copy_rope_seed42.yaml"
        ),
    },
    {
        "config": (
            PROJECT_ROOT
            / "configs"
            / "multiseed"
            / "reverse_learned_seed42.yaml"
        ),
    },
]


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
    "rope": "RoPE",
}

ENCODING_COLORS = {
    "learned": "#1F4E79",
    "sinusoidal": "#2E6F75",
    "rope": "#6B5F7A",
}

ENCODING_MARKERS = {
    "learned": "o",
    "sinusoidal": "s",
    "rope": "P",
}

ENCODING_LINESTYLES = {
    "learned": "-",
    "sinusoidal": "--",
    "rope": (0, (4, 1, 1, 1)),
}


UNIFORM_TOKEN_ERROR_REFERENCE = {
    "addition": 90.0,
    "copy": 99.0,
    "reverse": 99.0,
}


ERROR_COLOURMAP = LinearSegmentedColormap.from_list(
    "thesis_error",
    [
        "#F7F9FB",
        "#D7E2E7",
        "#A8C0CA",
        "#6F96A7",
        "#274C5A",
    ],
)


CARRY_STYLES = {
    "No carry-in": {
        "color": "#1F4E79",
        "marker": "o",
        "linestyle": "-",
    },
    "Carry-in": {
        "color": "#2E6F75",
        "marker": "s",
        "linestyle": "--",
    },
    "Leading carry digit": {
        "color": "#6B5F7A",
        "marker": "^",
        "linestyle": ":",
    },
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
        "xtick.labelsize": 9.0,
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
        alpha=0.70,
    )


def save_figure(
    figure,
    output_stem,
):
    figure.savefig(
        OUTPUT_DIR / f"{output_stem}.png",
        bbox_inches="tight",
        facecolor="white",
    )

    figure.savefig(
        OUTPUT_DIR / f"{output_stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(figure)


def choose_device():

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def build_model(
    config,
    device,
):

    model = TransformerModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        feedforward_dim=config["feedforward_dim"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        positional_encoding=config["positional_encoding"],
    )

    task = config["task"]
    train_length = config["train_length"]
    experiment_name = config["experiment_name"]

    checkpoint_path = (
        PROJECT_ROOT
        / config["checkpoint_dir"]
        / (
            f"{task}_train{train_length}_"
            f"{experiment_name}.pt"
        )
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, checkpoint_path


def expected_results_path(config):

    return (
        RERUN_RESULTS_DIR
        / (
            f"{config['task']}_"
            f"train{config['train_length']}_"
            f"{config['experiment_name']}_results.csv"
        )
    )


def load_expected_results(config):

    path = expected_results_path(config)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing validation result CSV: {path}"
        )

    return pd.read_csv(path)


def normalised_position_bins(number_of_tokens):

    relative_positions = np.arange(number_of_tokens)

    bins = (
        relative_positions
        * NUMBER_OF_POSITION_BINS
        // number_of_tokens
    )

    return np.minimum(
        bins,
        NUMBER_OF_POSITION_BINS - 1,
    ).astype(int)


def addition_carry_labels(
    input_tokens,
    digit_length,
    valid_target_length,
):

    first_operand = input_tokens[:digit_length]

    second_operand = input_tokens[
        digit_length + 1:
        digit_length + 1 + digit_length
    ]

    carry_in = [0] * digit_length
    carry = 0

    for column in range(
        digit_length - 1,
        -1,
        -1,
    ):
        carry_in[column] = carry

        total = (
            first_operand[column]
            + second_operand[column]
            + carry
        )

        carry = total // 10

    leading_offset = (
        valid_target_length - digit_length
    )

    labels = []

    for relative_position in range(
        valid_target_length
    ):
        if (
            leading_offset == 1
            and relative_position == 0
        ):
            labels.append("Leading carry digit")
            continue

        operand_column = (
            relative_position - leading_offset
        )

        if carry_in[operand_column] == 1:
            labels.append("Carry-in")
        else:
            labels.append("No carry-in")

    return labels


def reverse_dependency_bins(number_of_tokens):

    output_positions = np.arange(
        number_of_tokens
    )

    source_positions = (
        number_of_tokens
        - 1
        - output_positions
    )

    if number_of_tokens <= 1:
        normalised_distance = np.zeros(
            number_of_tokens
        )
    else:
        normalised_distance = (
            np.abs(
                source_positions
                - output_positions
            )
            / (number_of_tokens - 1)
        )

    bins = (
        normalised_distance
        * NUMBER_OF_POSITION_BINS
    ).astype(int)

    return np.minimum(
        bins,
        NUMBER_OF_POSITION_BINS - 1,
    )


def validate_aggregate_metrics(
    expected_results,
    test_length,
    token_accuracy,
    exact_accuracy,
    config,
):

    expected_row = expected_results[
        expected_results["Test Length"]
        == test_length
    ]

    if expected_row.empty:
        raise ValueError(
            "No expected result for "
            f"{config['task']} "
            f"{config['positional_encoding']} "
            f"length {test_length}"
        )

    expected_row = expected_row.iloc[0]

    expected_token = float(
        expected_row["Token Accuracy"]
    )

    expected_exact = float(
        expected_row["Exact Match Accuracy"]
    )

    token_matches = np.isclose(
        token_accuracy,
        expected_token,
        atol=1e-9,
        rtol=0.0,
    )

    exact_matches = np.isclose(
        exact_accuracy,
        expected_exact,
        atol=1e-9,
        rtol=0.0,
    )

    if not token_matches or not exact_matches:
        raise RuntimeError(
            "\nPrediction validation failed.\n"
            f"Task: {config['task']}\n"
            f"Encoding: {config['positional_encoding']}\n"
            f"Test length: {test_length}\n"
            f"Computed token accuracy: {token_accuracy}\n"
            f"Expected token accuracy: {expected_token}\n"
            f"Computed exact accuracy: {exact_accuracy}\n"
            f"Expected exact accuracy: {expected_exact}\n"
        )


def evaluate_experiment(
    config_path,
    device,
):

    config = load_config(config_path)

    task = config["task"]
    encoding = config["positional_encoding"]
    train_length = int(config["train_length"])
    test_lengths = [
        int(length)
        for length in config["test_lengths"]
    ]

    if int(config.get("seed", SEED)) != SEED:
        raise ValueError(
            "This diagnostic is restricted to seed 42."
        )

    model, checkpoint_path = build_model(
        config,
        device,
    )

    expected_results = load_expected_results(
        config
    )

    print(
        "\nEvaluating position errors for "
        f"{TASK_LABELS[task]} / "
        f"{ENCODING_LABELS[encoding]}"
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    position_rows = []
    overall_rows = []
    carry_rows = []
    dependency_rows = []

    pad_token = config.get("pad_token")
    data_dir = PROJECT_ROOT / config["data_dir"]

    for test_length in test_lengths:
        print(
            f"  Test length: {test_length}"
        )

        test_path = (
            data_dir
            / f"test_{test_length}.jsonl"
        )

        if not test_path.exists():
            raise FileNotFoundError(
                f"Missing test dataset: {test_path}"
            )

        loader = get_dataloader(
            test_path,
            batch_size=config["batch_size"],
            shuffle=False,
        )

        position_error_counts = np.zeros(
            NUMBER_OF_POSITION_BINS,
            dtype=np.int64,
        )

        position_token_counts = np.zeros(
            NUMBER_OF_POSITION_BINS,
            dtype=np.int64,
        )

        carry_error_counts = defaultdict(int)
        carry_token_counts = defaultdict(int)

        dependency_error_counts = np.zeros(
            NUMBER_OF_POSITION_BINS,
            dtype=np.int64,
        )

        dependency_token_counts = np.zeros(
            NUMBER_OF_POSITION_BINS,
            dtype=np.int64,
        )

        total_errors = 0
        total_tokens = 0
        exact_matches = 0
        total_sequences = 0

        with torch.no_grad():
            for inputs_cpu, targets_cpu in loader:
                inputs = inputs_cpu.to(device)
                targets = targets_cpu.to(device)

                logits = model(inputs)

                predictions = (
                    logits.argmax(dim=-1).cpu()
                )

                if pad_token is None:
                    valid_mask = torch.ones_like(
                        targets_cpu,
                        dtype=torch.bool,
                    )
                else:
                    valid_mask = (
                        targets_cpu != pad_token
                    )

                error_mask = (
                    predictions != targets_cpu
                ) & valid_mask

                total_errors += int(
                    error_mask.sum().item()
                )

                total_tokens += int(
                    valid_mask.sum().item()
                )

                sequence_correct = (
                    (predictions == targets_cpu)
                    | ~valid_mask
                ).all(dim=1)

                exact_matches += int(
                    sequence_correct.sum().item()
                )

                total_sequences += int(
                    targets_cpu.size(0)
                )

                for sample_index in range(
                    targets_cpu.size(0)
                ):
                    valid_positions = (
                        torch.nonzero(
                            valid_mask[sample_index],
                            as_tuple=False,
                        )
                        .flatten()
                    )

                    number_of_valid_tokens = int(
                        valid_positions.numel()
                    )

                    if number_of_valid_tokens == 0:
                        continue

                    sample_errors = (
                        error_mask[
                            sample_index,
                            valid_positions,
                        ]
                        .numpy()
                        .astype(np.int64)
                    )

                    position_bins = (
                        normalised_position_bins(
                            number_of_valid_tokens
                        )
                    )

                    np.add.at(
                        position_error_counts,
                        position_bins,
                        sample_errors,
                    )

                    np.add.at(
                        position_token_counts,
                        position_bins,
                        1,
                    )

                    if task == "addition":
                        input_tokens = (
                            inputs_cpu[sample_index]
                            .tolist()
                        )

                        carry_labels = (
                            addition_carry_labels(
                                input_tokens,
                                test_length,
                                number_of_valid_tokens,
                            )
                        )

                        for label, error in zip(
                            carry_labels,
                            sample_errors,
                        ):
                            carry_token_counts[
                                label
                            ] += 1

                            carry_error_counts[
                                label
                            ] += int(error)

                    if task == "reverse":
                        dependency_bins = (
                            reverse_dependency_bins(
                                number_of_valid_tokens
                            )
                        )

                        np.add.at(
                            dependency_error_counts,
                            dependency_bins,
                            sample_errors,
                        )

                        np.add.at(
                            dependency_token_counts,
                            dependency_bins,
                            1,
                        )

        token_accuracy = (
            100.0
            * (total_tokens - total_errors)
            / total_tokens
        )

        token_error = (
            100.0 - token_accuracy
        )

        exact_accuracy = (
            100.0
            * exact_matches
            / total_sequences
        )

        exact_error = (
            100.0 - exact_accuracy
        )

        validate_aggregate_metrics(
            expected_results,
            test_length,
            token_accuracy,
            exact_accuracy,
            config,
        )

        overall_rows.append({
            "Task": task,
            "Task Label": TASK_LABELS[task],
            "Positional Encoding": encoding,
            "Encoding Label": (
                ENCODING_LABELS[encoding]
            ),
            "Seed": SEED,
            "Train Length": train_length,
            "Test Length": test_length,
            "Token Accuracy (%)": token_accuracy,
            "Token Error (%)": token_error,
            "Exact-Match Accuracy (%)": (
                exact_accuracy
            ),
            "Exact-Match Error (%)": exact_error,
            "Checkpoint": str(checkpoint_path),
            "Diagnostic Scope": (
                "Representative seed-42 checkpoint"
            ),
        })

        for position_bin in range(
            NUMBER_OF_POSITION_BINS
        ):
            count = int(
                position_token_counts[
                    position_bin
                ]
            )

            errors = int(
                position_error_counts[
                    position_bin
                ]
            )

            error_rate = (
                100.0 * errors / count
                if count > 0
                else np.nan
            )

            position_rows.append({
                "Task": task,
                "Task Label": TASK_LABELS[task],
                "Positional Encoding": encoding,
                "Encoding Label": (
                    ENCODING_LABELS[encoding]
                ),
                "Seed": SEED,
                "Train Length": train_length,
                "Test Length": test_length,
                "Position Bin": position_bin,
                "Bin Start (%)": (
                    10 * position_bin
                ),
                "Bin End (%)": (
                    10 * (position_bin + 1)
                ),
                "Token Count": count,
                "Error Count": errors,
                "Error Rate (%)": error_rate,
            })

        if task == "addition":
            for carry_label in CARRY_STYLES:
                count = carry_token_counts[
                    carry_label
                ]

                errors = carry_error_counts[
                    carry_label
                ]

                error_rate = (
                    100.0 * errors / count
                    if count > 0
                    else np.nan
                )

                carry_rows.append({
                    "Task": task,
                    "Positional Encoding": encoding,
                    "Encoding Label": (
                        ENCODING_LABELS[encoding]
                    ),
                    "Seed": SEED,
                    "Train Length": train_length,
                    "Test Length": test_length,
                    "Carry Condition": carry_label,
                    "Token Count": count,
                    "Error Count": errors,
                    "Error Rate (%)": error_rate,
                })

        if task == "reverse":
            for distance_bin in range(
                NUMBER_OF_POSITION_BINS
            ):
                count = int(
                    dependency_token_counts[
                        distance_bin
                    ]
                )

                errors = int(
                    dependency_error_counts[
                        distance_bin
                    ]
                )

                error_rate = (
                    100.0 * errors / count
                    if count > 0
                    else np.nan
                )

                dependency_rows.append({
                    "Task": task,
                    "Positional Encoding": encoding,
                    "Encoding Label": (
                        ENCODING_LABELS[encoding]
                    ),
                    "Seed": SEED,
                    "Train Length": train_length,
                    "Test Length": test_length,
                    "Distance Bin": distance_bin,
                    "Distance Start (%)": (
                        10 * distance_bin
                    ),
                    "Distance End (%)": (
                        10 * (distance_bin + 1)
                    ),
                    "Token Count": count,
                    "Error Count": errors,
                    "Error Rate (%)": error_rate,
                })

        print(
            "    Validated: "
            f"token error={token_error:.2f}%, "
            f"exact error={exact_error:.2f}%"
        )

    return (
        position_rows,
        overall_rows,
        carry_rows,
        dependency_rows,
    )


def task_encodings(position_data, task):

    available = set(
        position_data.loc[
            position_data["Task"] == task,
            "Positional Encoding",
        ]
    )

    return [
        encoding
        for encoding in [
            "learned",
            "sinusoidal",
            "rope",
        ]
        if encoding in available
    ]


def save_position_heatmap(
    position_data,
    task,
):

    encodings = task_encodings(
        position_data,
        task,
    )

    figure, axes = plt.subplots(
        1,
        len(encodings),
        figsize=(
            5.3 * len(encodings),
            5.0,
        ),
        squeeze=False,
        constrained_layout=True,
    )

    axes = axes.ravel()
    image = None

    for axis, encoding in zip(
        axes,
        encodings,
    ):
        frame = position_data[
            (position_data["Task"] == task)
            & (
                position_data[
                    "Positional Encoding"
                ]
                == encoding
            )
        ]

        matrix = (
            frame.pivot(
                index="Test Length",
                columns="Position Bin",
                values="Error Rate (%)",
            )
            .sort_index()
            .reindex(
                columns=range(
                    NUMBER_OF_POSITION_BINS
                )
            )
        )

        image = axis.imshow(
            matrix.to_numpy(),
            aspect="auto",
            interpolation="nearest",
            cmap=ERROR_COLOURMAP,
            vmin=0.0,
            vmax=100.0,
        )

        axis.set_title(
            ENCODING_LABELS[encoding]
        )

        axis.set_xticks(
            range(NUMBER_OF_POSITION_BINS)
        )

        axis.set_xticklabels(
            [
                f"{10 * index}–"
                f"{10 * (index + 1)}"
                for index in range(
                    NUMBER_OF_POSITION_BINS
                )
            ],
            rotation=45,
            ha="right",
        )

        training_length = int(
            frame["Train Length"].iloc[0]
        )

        y_labels = []

        for length in matrix.index:
            if int(length) == training_length:
                y_labels.append(
                    f"{int(length)} (train)"
                )
            else:
                y_labels.append(str(int(length)))

        axis.set_yticks(
            range(len(matrix.index))
        )

        axis.set_yticklabels(y_labels)

        axis.set_xlabel(
            "Relative target position (%)"
        )

        axis.set_ylabel(
            "Test sequence length"
        )

        axis.set_xticks(
            np.arange(
                -0.5,
                NUMBER_OF_POSITION_BINS,
                1,
            ),
            minor=True,
        )

        axis.set_yticks(
            np.arange(
                -0.5,
                len(matrix.index),
                1,
            ),
            minor=True,
        )

        axis.grid(
            which="minor",
            color="white",
            linewidth=0.8,
        )

        axis.tick_params(
            which="minor",
            bottom=False,
            left=False,
        )

    figure.suptitle(
        f"{TASK_LABELS[task]}: "
        "position-wise token error "
        "(representative seed 42)",
        fontsize=12.5,
    )

    colourbar = figure.colorbar(
        image,
        ax=axes.tolist(),
        shrink=0.86,
        pad=0.025,
    )

    colourbar.set_label(
        "Token error rate (%)"
    )

    save_figure(
        figure,
        f"{task}_position_error_heatmap",
    )


def save_train_vs_first_ood_plot(
    position_data,
    task,
):

    encodings = task_encodings(
        position_data,
        task,
    )

    task_frame = position_data[
        position_data["Task"] == task
    ]

    train_length = int(
        task_frame["Train Length"].iloc[0]
    )

    first_ood_length = int(
        task_frame.loc[
            task_frame["Test Length"]
            > train_length,
            "Test Length",
        ].min()
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.5),
        sharey=True,
        constrained_layout=True,
    )

    panel_settings = [
        (
            axes[0],
            train_length,
            f"Training length ({train_length})",
        ),
        (
            axes[1],
            first_ood_length,
            (
                "First unseen length "
                f"({first_ood_length})"
            ),
        ),
    ]

    x_values = (
        np.arange(NUMBER_OF_POSITION_BINS)
        * 10
        + 5
    )

    for axis, length, title in panel_settings:
        for encoding in encodings:
            frame = task_frame[
                (
                    task_frame[
                        "Positional Encoding"
                    ]
                    == encoding
                )
                & (
                    task_frame["Test Length"]
                    == length
                )
            ].sort_values("Position Bin")

            axis.plot(
                x_values,
                frame["Error Rate (%)"],
                color=ENCODING_COLORS[
                    encoding
                ],
                marker=ENCODING_MARKERS[
                    encoding
                ],
                linestyle=ENCODING_LINESTYLES[
                    encoding
                ],
                markersize=5.5,
                label=ENCODING_LABELS[
                    encoding
                ],
            )

        axis.set_title(title)
        axis.set_xlabel(
            "Relative target position (%)"
        )
        axis.set_xlim(0, 100)
        axis.set_ylim(-3, 103)
        axis.set_xticks(
            [5, 25, 45, 65, 85, 95]
        )
        axis.set_xticklabels(
            [
                "Start",
                "25",
                "45",
                "65",
                "85",
                "End",
            ]
        )

        style_axis(axis)

    axes[0].set_ylabel(
        "Token error rate (%)"
    )

    axes[1].axhline(
        UNIFORM_TOKEN_ERROR_REFERENCE[
            task
        ],
        color="#707070",
        linestyle=":",
        linewidth=1.4,
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
            label=ENCODING_LABELS[
                encoding
            ],
        )
        for encoding in encodings
    ]

    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#707070",
            linestyle=":",
            label=(
                "Uniform valid-token "
                "error reference"
            ),
        )
    )

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=len(legend_handles),
        frameon=False,
    )

    figure.suptitle(
        f"{TASK_LABELS[task]}: "
        "where token errors occur "
        "(representative seed 42)",
        fontsize=12.5,
    )

    save_figure(
        figure,
        (
            f"{task}_"
            "train_vs_first_ood_"
            "position_error"
        ),
    )


def save_addition_carry_plot(carry_data):

    encodings = [
        encoding
        for encoding in [
            "learned",
            "sinusoidal",
        ]
        if encoding
        in set(
            carry_data[
                "Positional Encoding"
            ]
        )
    ]

    figure, axes = plt.subplots(
        1,
        len(encodings),
        figsize=(
            5.4 * len(encodings),
            4.6,
        ),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )

    axes = axes.ravel()

    for axis, encoding in zip(
        axes,
        encodings,
    ):
        encoding_frame = carry_data[
            carry_data[
                "Positional Encoding"
            ]
            == encoding
        ]

        lengths = sorted(
            encoding_frame[
                "Test Length"
            ].unique()
        )

        x_values = np.arange(
            len(lengths)
        )

        for carry_condition, style in (
            CARRY_STYLES.items()
        ):
            frame = (
                encoding_frame[
                    encoding_frame[
                        "Carry Condition"
                    ]
                    == carry_condition
                ]
                .set_index("Test Length")
                .reindex(lengths)
            )

            axis.plot(
                x_values,
                frame["Error Rate (%)"],
                label=carry_condition,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                markersize=5.5,
            )

        axis.axvspan(
            0.5,
            len(lengths) - 0.5,
            color="#F2F4F6",
            alpha=0.65,
            zorder=0,
        )

        axis.set_title(
            ENCODING_LABELS[encoding]
        )

        axis.set_xlabel(
            "Test digit length"
        )

        axis.set_xticks(x_values)
        axis.set_xticklabels(lengths)
        axis.set_ylim(-3, 103)

        style_axis(axis)

    axes[0].set_ylabel(
        "Digit error rate (%)"
    )

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            label=label,
        )
        for label, style in (
            CARRY_STYLES.items()
        )
    ]

    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        frameon=False,
    )

    figure.suptitle(
        "Addition: error by carry condition "
        "(representative seed 42)",
        fontsize=12.5,
    )

    save_figure(
        figure,
        "addition_carry_condition_error",
    )


def save_reverse_dependency_heatmap(
    dependency_data
):
    matrix = (
        dependency_data.pivot(
            index="Test Length",
            columns="Distance Bin",
            values="Error Rate (%)",
        )
        .sort_index()
        .reindex(
            columns=range(
                NUMBER_OF_POSITION_BINS
            )
        )
    )

    figure, axis = plt.subplots(
        figsize=(6.8, 4.8),
        constrained_layout=True,
    )

    image = axis.imshow(
        matrix.to_numpy(),
        aspect="auto",
        interpolation="nearest",
        cmap=ERROR_COLOURMAP,
        vmin=0.0,
        vmax=100.0,
    )

    axis.set_xticks(
        range(NUMBER_OF_POSITION_BINS)
    )

    axis.set_xticklabels(
        [
            f"{10 * index}–"
            f"{10 * (index + 1)}"
            for index in range(
                NUMBER_OF_POSITION_BINS
            )
        ],
        rotation=45,
        ha="right",
    )

    train_length = int(
        dependency_data[
            "Train Length"
        ].iloc[0]
    )

    y_labels = [
        (
            f"{int(length)} (train)"
            if int(length) == train_length
            else str(int(length))
        )
        for length in matrix.index
    ]

    axis.set_yticks(
        range(len(matrix.index))
    )

    axis.set_yticklabels(y_labels)

    axis.set_xlabel(
        "Normalised input-output "
        "dependency distance (%)"
    )

    axis.set_ylabel(
        "Test sequence length"
    )

    axis.set_title(
        "Reverse: error by dependency distance "
        "(Learned, seed 42)"
    )

    axis.set_xticks(
        np.arange(
            -0.5,
            NUMBER_OF_POSITION_BINS,
            1,
        ),
        minor=True,
    )

    axis.set_yticks(
        np.arange(
            -0.5,
            len(matrix.index),
            1,
        ),
        minor=True,
    )

    axis.grid(
        which="minor",
        color="white",
        linewidth=0.8,
    )

    axis.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    colourbar = figure.colorbar(
        image,
        ax=axis,
        shrink=0.88,
        pad=0.03,
    )

    colourbar.set_label(
        "Token error rate (%)"
    )

    save_figure(
        figure,
        "reverse_dependency_distance_error",
    )


def classify_position_pattern(
    overall_error,
    start_error,
    middle_error,
    end_error,
    error_range,
    uniform_reference,
):

    reference_difference = abs(
        overall_error
        - uniform_reference
    )

    if (
        reference_difference <= 5.0
        and error_range <= 12.0
    ):
        return "Uniform near-chance collapse"

    if end_error - start_error >= 15.0:
        return "Errors concentrated toward output end"

    if start_error - end_error >= 15.0:
        return "Errors concentrated toward output start"

    boundary_error = (
        start_error + end_error
    ) / 2.0

    if boundary_error - middle_error >= 15.0:
        return "Boundary-concentrated failure"

    if reference_difference <= 5.0:
        return "Distributed near-chance collapse"

    if overall_error >= 80.0:
        return "Distributed high-error collapse"

    return "Structured positional degradation"


def create_failure_summary(
    position_data,
    overall_data,
):

    rows = []

    group_columns = [
        "Task",
        "Positional Encoding",
    ]

    for (
        task,
        encoding,
    ), overall_frame in overall_data.groupby(
        group_columns
    ):
        overall_frame = (
            overall_frame.sort_values(
                "Test Length"
            )
        )

        train_length = int(
            overall_frame[
                "Train Length"
            ].iloc[0]
        )

        first_ood_length = int(
            overall_frame.loc[
                overall_frame[
                    "Test Length"
                ]
                > train_length,
                "Test Length",
            ].min()
        )

        ood_overall = overall_frame[
            overall_frame["Test Length"]
            == first_ood_length
        ].iloc[0]

        profile = (
            position_data[
                (
                    position_data["Task"]
                    == task
                )
                & (
                    position_data[
                        "Positional Encoding"
                    ]
                    == encoding
                )
                & (
                    position_data[
                        "Test Length"
                    ]
                    == first_ood_length
                )
            ]
            .sort_values("Position Bin")
        )

        error_values = (
            profile["Error Rate (%)"]
            .to_numpy()
        )

        start_error = float(
            error_values[0]
        )

        middle_error = float(
            np.nanmean(
                error_values[4:6]
            )
        )

        end_error = float(
            error_values[-1]
        )

        error_range = float(
            np.nanmax(error_values)
            - np.nanmin(error_values)
        )

        overall_error = float(
            ood_overall[
                "Token Error (%)"
            ]
        )

        failure_pattern = (
            classify_position_pattern(
                overall_error,
                start_error,
                middle_error,
                end_error,
                error_range,
                UNIFORM_TOKEN_ERROR_REFERENCE[
                    task
                ],
            )
        )

        rows.append({
            "Task": TASK_LABELS[task],
            "Positional Encoding": (
                ENCODING_LABELS[encoding]
            ),
            "Seed": SEED,
            "Train Length": train_length,
            "First Unseen Length": (
                first_ood_length
            ),
            "Overall Token Error (%)": (
                overall_error
            ),
            "Start-Bin Error (%)": (
                start_error
            ),
            "Middle-Bin Error (%)": (
                middle_error
            ),
            "End-Bin Error (%)": (
                end_error
            ),
            "Position Error Range "
            "(percentage points)": (
                error_range
            ),
            "Uniform Error Reference (%)": (
                UNIFORM_TOKEN_ERROR_REFERENCE[
                    task
                ]
            ),
            "Failure Pattern": failure_pattern,
            "Diagnostic Scope": (
                "Representative seed-42 "
                "checkpoint; multi-seed eligibility "
                "is reported separately"
            ),
        })

    return pd.DataFrame(rows)


def main():
    set_report_style()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = choose_device()

    print(f"Using device: {device}")

    if device.type != "cuda":
        print(
            "Warning: CUDA is not active. "
            "This diagnostic will be slower."
        )

    all_position_rows = []
    all_overall_rows = []
    all_carry_rows = []
    all_dependency_rows = []

    for experiment in EXPERIMENTS:
        (
            position_rows,
            overall_rows,
            carry_rows,
            dependency_rows,
        ) = evaluate_experiment(
            experiment["config"],
            device,
        )

        all_position_rows.extend(
            position_rows
        )

        all_overall_rows.extend(
            overall_rows
        )

        all_carry_rows.extend(
            carry_rows
        )

        all_dependency_rows.extend(
            dependency_rows
        )

    position_data = pd.DataFrame(
        all_position_rows
    )

    overall_data = pd.DataFrame(
        all_overall_rows
    )

    carry_data = pd.DataFrame(
        all_carry_rows
    )

    dependency_data = pd.DataFrame(
        all_dependency_rows
    )

    position_data.to_csv(
        OUTPUT_DIR
        / "position_error_by_bin.csv",
        index=False,
    )

    overall_data.to_csv(
        OUTPUT_DIR
        / "position_diagnostic_aggregate_check.csv",
        index=False,
    )

    carry_data.to_csv(
        OUTPUT_DIR
        / "addition_carry_condition_error.csv",
        index=False,
    )

    dependency_data.to_csv(
        OUTPUT_DIR
        / "reverse_dependency_distance_error.csv",
        index=False,
    )

    failure_summary = (
        create_failure_summary(
            position_data,
            overall_data,
        )
    )

    failure_summary.to_csv(
        OUTPUT_DIR
        / "position_failure_summary.csv",
        index=False,
    )

    for task in TASK_ORDER:
        save_position_heatmap(
            position_data,
            task,
        )

        save_train_vs_first_ood_plot(
            position_data,
            task,
        )

    save_addition_carry_plot(
        carry_data
    )

    save_reverse_dependency_heatmap(
        dependency_data
    )

    print(
        "\nSaved position diagnostics to:"
    )

    print(OUTPUT_DIR)

    print(
        "\nAggregate metrics matched all five "
        "independent rerun CSVs."
    )


if __name__ == "__main__":
    main()