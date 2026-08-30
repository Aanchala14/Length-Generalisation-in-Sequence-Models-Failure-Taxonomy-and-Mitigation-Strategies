# Length Generalisation in Transformer-Based Sequence Models

This repository contains the implementation, experiment configurations,
results, diagnostics, and report figures for the MSc dissertation:

**Length Generalisation in Transformer-Based Sequence Models: Failure Taxonomy
and Mitigation Strategies**

The project investigates whether compact Transformer models trained on short
sequences learn computations that transfer to longer sequences. It moves
beyond asking only whether accuracy decreases by separating training
underfitting, seed instability, extrapolation collapse, and different spatial
patterns of token error.

## Research Aim

The primary research question is:

> When Transformer models are trained on short algorithmic sequences, which
> distinguishable failure behaviours occur when they are evaluated on longer
> sequences, and do targeted training interventions improve extrapolation?

The experiments follow a controlled train-short/test-long design. Models are
first required to learn the training task reliably before their longer-length
performance is interpreted as evidence about length extrapolation.

## Tasks

The main synthetic benchmark contains three deterministic sequence tasks.

| Task | Training length | Evaluation lengths | Capability examined |
|---|---:|---|---|
| Addition | 16 | 16, 32, 64, 128, 256, 512, 1024 | Digit computation and carry propagation |
| Delayed Copy | 128 | 128, 256, 512, 1024 | Retention and retrieval after a delay |
| Reverse | 128 | 128, 256, 512, 1024 | Position-dependent sequence remapping |

Associative recall was implemented during development but excluded from the
main comparison because it was not learned reliably enough at the training
length to provide a valid extrapolation baseline.

An exploratory real-world extension uses the SelfRegulationSCP2 multivariate
time-series classification dataset.

## Experimental Design

Five positional encoding conditions are compared:

- learned absolute positional embeddings;
- sinusoidal positional encodings;
- no explicit positional encoding (NoPE);
- Attention with Linear Biases (ALiBi);
- Rotary Position Embedding (RoPE).

The main synthetic benchmark contains:

```text
3 tasks × 5 positional encoding conditions × 3 seeds = 45 models
```

The seeds are `42`, `123`, and `2024`.

The real-world extension contains:

```text
1 dataset × 5 positional encoding conditions × 3 seeds = 15 models
```

Targeted mitigation experiments use representative seed-42 configurations and
matched seed-42 baselines.

## Evaluation Framework

Synthetic models are evaluated using:

- token accuracy;
- token-error percentage;
- exact-match accuracy;
- position-wise token error;
- task-specific carry and dependency-distance diagnostics;
- descriptive attention statistics.

A task-encoding baseline is eligible for extrapolation analysis only when all
three seeds achieve at least `90%` exact-match accuracy at the training length.

The collapse threshold is defined as exact-match accuracy below `10%`. The
first failure length is the first evaluated unseen length below this threshold.

Position-wise and task-specific diagnostics use representative seed-42
checkpoints. Multi-seed baseline eligibility is reported separately.

## Main Findings

Five task-encoding configurations satisfied the multi-seed baseline-eligibility
criterion:

- Addition with learned positional embeddings;
- Addition with sinusoidal positional encodings;
- Delayed Copy with sinusoidal positional encodings;
- Delayed Copy with RoPE;
- Reverse with learned positional embeddings.

All five fell to `0%` exact-match accuracy at the first evaluated unseen
length.

At that first unseen length:

- Addition produced approximately `86.9–88.0%` token error and retained a
  modest position-wise error gradient.
- Delayed Copy produced approximately `99%` token error distributed almost
  uniformly across output positions.
- Reverse produced approximately `98.8%` token error distributed almost
  uniformly across output positions.
- Addition error was not primarily concentrated on digits receiving an
  incoming carry.
- Reverse error was not concentrated at one dependency-distance range.

Mixed-length variants underfit in the evaluated settings. Curriculum and
randomised-padded variants that preserved training-length performance produced
no observed exact-match improvement at unseen lengths.

The SelfRegulationSCP2 extension produced a comparatively flat but weak
classification curve of approximately `49–56%` mean accuracy. Because
training-prefix performance was already close to the uniform two-class
reference, this experiment is treated as exploratory rather than evidence of
successful real-world length generalisation.

## Failure Taxonomy

The final interpretation distinguishes five empirical outcomes:

1. **Training-length underfitting**  
   The model does not reliably learn the original training task.

2. **Seed instability**  
   Training success changes substantially across random seeds.

3. **Immediate extrapolation collapse**  
   An eligible baseline falls below the collapse threshold at the first
   evaluated unseen length.

4. **Position-dependent error structure**  
   Token errors may be nearly uniform or retain a systematic positional
   gradient after exact-match collapse.

5. **No observed improvement under tested mitigation**  
   A tested intervention either underfits or preserves training performance
   without improving unseen-length exact match.

Attention entropy, attention distance, and local-attention ratio are retained
as descriptive supporting evidence. They are not treated as independent
failure categories or causal explanations.

## Repository Structure

```text
.
├── configs/
│   ├── multiseed/          Multi-seed synthetic baseline configurations
│   ├── mitigation/         Mixed, curriculum, padded, and control configs
│   └── realworld/          SelfRegulationSCP2 configurations
│
├── data/
│   └── synthetic/          Generated task datasets
│
├── src/
│   ├── tasks/              Synthetic task definitions and generation
│   ├── data/               Dataset loaders and tokenisation
│   ├── models/             Transformer and positional encoding modules
│   ├── training/           Training, evaluation, analysis, and plotting
│   ├── evaluation/         Evaluation support code
│   ├── analysis/           Analysis support code
│   └── utils/              Configuration, logging, and seed utilities
│
├── scripts/                Configuration and experiment helper scripts
├── tests/                  Unit and implementation tests
├── notebooks/              Exploratory notebooks
├── experiments/            Experiment notes and supporting material
│
├── outputs/
│   ├── checkpoints/        Saved model checkpoints
│   ├── results/            Raw evaluation CSV files
│   ├── analysis/           Derived diagnostic summaries
│   ├── plots/final/        Final report-ready PNG and PDF figures
│   └── logs/               Training and evaluation logs
│
├── position_diagnostics_aws/
│   └── final/              Synced AWS position-diagnostic outputs
│
├── latex/
│   ├── Bhongade_ravindra_Aanchala_mscthesis.tex
│   ├── Bhongade_Ravindra_Aanchala_mscthesis.pdf
│   └── references.bib
│
├── README.md
└── requirements.txt
```

## Environment Setup

Python 3.11 or later is recommended. A CUDA-enabled GPU is useful for the full
experiment matrix and checkpoint-based position diagnostics, although smaller
experiments and plotting scripts can run locally.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies include PyTorch, NumPy, pandas, Matplotlib, PyYAML,
scikit-learn, and Aeon.

For AWS GPU experiments, install a PyTorch build compatible with the CUDA
version available on the instance.

## Synthetic Experiment Workflow

Each experiment is controlled by a YAML configuration file. The standard
workflow is:

1. generate the configured datasets;
2. train the model;
3. load the saved checkpoint;
4. evaluate every configured test length;
5. save the results as CSV;
6. create aggregated summaries and figures.

Run all commands from the repository root with `PYTHONPATH=.`.

### Example: Addition, learned encoding, seed 42

```bash
PYTHONPATH=. python -m src.tasks.generate_dataset \
  --config configs/multiseed/addition_learned_seed42.yaml

PYTHONPATH=. python -m src.training.train \
  --config configs/multiseed/addition_learned_seed42.yaml

PYTHONPATH=. python -m src.training.evaluate \
  --config configs/multiseed/addition_learned_seed42.yaml
```

The configuration determines the task, train and test lengths, model
hyperparameters, positional encoding, seed, data directory, checkpoint
directory, and results directory.

The main synthetic result CSV files are stored in:

```text
outputs/results/multiseed_results/
```

## Baseline Eligibility and Extrapolation Plots

Generate the multi-seed baseline-eligibility analysis with:

```bash
PYTHONPATH=. python -m src.training.plot_baseline_diagnostics
```

This produces:

- training-fit results by task, encoding, and seed;
- the baseline-eligibility summary;
- extrapolation curves for eligible baselines;
- error-decomposition summaries.

Outputs are written to:

```text
outputs/plots/final/
```

## Baseline-versus-Mitigation Figures

Generate the matched baseline and mitigation figures with:

```bash
PYTHONPATH=. python -m src.training.plot_final_report_figures
```

The plotted comparisons include:

- Addition learned baseline versus mixed-length variants;
- Delayed Copy sinusoidal baseline versus control, mixed-length, curriculum,
  and randomised-padded variants;
- Delayed Copy RoPE baseline versus curriculum and randomised-padded variants;
- Reverse learned baseline versus curriculum training.

The script also produces plot-data CSV files and summary tables in:

```text
outputs/plots/final/
```

## Position-Wise Diagnostics

The position-diagnostic analysis uses the five eligible representative
seed-42 checkpoints. It computes:

- position-wise token error;
- training-length versus first-unseen-length profiles;
- full-length error heatmaps;
- Addition carry-conditioned error;
- Reverse dependency-distance error;
- aggregate checks against independently generated evaluation CSV files.

The required independent evaluation CSV files are expected in:

```text
outputs/results/position_diagnostic_reruns/
```

Run:

```bash
PYTHONPATH=. python -m src.training.plot_position_diagnostics
```

This analysis loads trained checkpoints and re-evaluates datasets, so a CUDA
GPU is recommended.

The main outputs are written to:

```text
outputs/plots/final/
```

The corrected Addition carry-condition figure can be regenerated with:

```bash
PYTHONPATH=. python -m src.training.replot_addition_carry_diagnostic
```

Synced AWS copies of the diagnostic plots and supporting CSV files are retained
in:

```text
position_diagnostics_aws/final/
```

## Attention Diagnostics

Attention diagnostics are calculated for selected checkpoints at the training
length, first unseen length, and longest evaluated length.

Example:

```bash
PYTHONPATH=. python -m src.training.analyze_attention \
  --config configs/multiseed/addition_learned_seed42.yaml \
  --length 16 \
  --max_batches 5
```

Repeat the command with the relevant diagnostic lengths and configurations.

Summarise the saved attention outputs with:

```bash
PYTHONPATH=. python -m src.training.summarise_attention
```

Attention data and summaries are stored in:

```text
outputs/analysis/attention/
outputs/analysis/attention_summary/
```

These statistics are descriptive and should not be interpreted as proof of a
causal failure mechanism.

## Mitigation Experiments

Mitigation configurations are stored in:

```text
configs/mitigation/
```

The tested interventions are:

- **mixed-length training**, which changes length exposure;
- **curriculum training**, which changes the optimisation schedule;
- **randomised padded training**, which changes the absolute positions occupied
  by meaningful tokens;
- a **single-length control**, which checks whether pipeline differences alone
  explain a mitigation result.

Train and evaluate a mitigation configuration using the same modules as the
baseline experiments:

```bash
PYTHONPATH=. python -m src.tasks.generate_dataset \
  --config configs/mitigation/copy_randomised_sinusoidal_seed42.yaml

PYTHONPATH=. python -m src.training.train \
  --config configs/mitigation/copy_randomised_sinusoidal_seed42.yaml

PYTHONPATH=. python -m src.training.evaluate \
  --config configs/mitigation/copy_randomised_sinusoidal_seed42.yaml
```

Summarise the available mitigation results with:

```bash
PYTHONPATH=. python -m src.training.summarise_mitigation_results
```

Mitigation outputs are stored in:

```text
outputs/results/mitigation_results/
outputs/checkpoints/mitigation_checkpoints/
outputs/analysis/mitigation_summary/
```

A mitigation that does not preserve training-length performance is classified
as underfitting rather than successful or unsuccessful extrapolation
mitigation.

## Real-World Extension

The real-world extension uses SelfRegulationSCP2 from the UEA multivariate
time-series archive.

Example:

```bash
PYTHONPATH=. python -m src.training.train_timeseries \
  --config configs/realworld/selfregulationscp2_sinusoidal_seed42.yaml

PYTHONPATH=. python -m src.training.evaluate_timeseries \
  --config configs/realworld/selfregulationscp2_sinusoidal_seed42.yaml
```

Generate the original aggregate analysis with:

```bash
PYTHONPATH=. python -m src.training.plot_realworld_results
```

Generate the consistently styled final report figure with:

```bash
PYTHONPATH=. python -m src.training.replot_realworld_results
```

Results and figures are stored in:

```text
outputs/results/realworld/
outputs/plots/realworld/
outputs/plots/final/selfregulationscp2_accuracy_mean_se.png
outputs/plots/final/selfregulationscp2_accuracy_mean_se.pdf
```

## Final Report Figures

The main report-ready figures are available as both PNG and PDF where
applicable in:

```text
outputs/plots/final/
```

They include:

```text
controlled_task_examples
baseline_training_fit_by_seed

addition_eligible_baseline_extrapolation
copy_eligible_baseline_extrapolation
reverse_eligible_baseline_extrapolation

addition_eligible_baseline_error_decomposition
copy_eligible_baseline_error_decomposition
reverse_eligible_baseline_error_decomposition

addition_train_vs_first_ood_position_error
copy_train_vs_first_ood_position_error
reverse_train_vs_first_ood_position_error

addition_position_error_heatmap
copy_position_error_heatmap
reverse_position_error_heatmap

addition_carry_condition_error
reverse_dependency_distance_error

addition_learned_baseline_vs_mitigation
copy_sinusoidal_baseline_vs_mitigation
copy_rope_baseline_vs_mitigation
reverse_learned_baseline_vs_mitigation

selfregulationscp2_accuracy_mean_se
```

Vector PDF versions should be preferred when including figures in LaTeX.

## Tests

Run the repository tests from the project root:

```bash
PYTHONPATH=. pytest -q
```

The tests cover task generation, configuration loading, datasets, data
loading, positional encodings, Transformer behaviour, and available device
support.

## Reproducibility Notes

- Run commands from the repository root.
- Use `PYTHONPATH=.` for module imports.
- Check each YAML file before running because it determines all output paths.
- Re-running an experiment with the same configuration may overwrite its
  checkpoint, result CSV, or plot.
- Baseline conclusions use seeds `42`, `123`, and `2024`.
- Detailed position, carry, dependency-distance, attention, and mitigation
  analyses use representative seed-42 checkpoints unless stated otherwise.
- The validation split is not used for early stopping or checkpoint selection.
- The same final-epoch checkpoint is evaluated at every configured test length.
- Padding targets are excluded consistently from relevant loss and metric
  calculations.
- CUDA-enabled AWS instances were used for the larger training and diagnostic
  runs.
- Raw CSV results are retained separately from derived summaries and final
  figures.

## Dissertation

The dissertation source, bibliography, and compiled report are stored in:

```text
latex/
```

The final report uses:

- raw results from `outputs/results/`;
- derived analysis from `outputs/analysis/`;
- position-diagnostic summaries from `position_diagnostics_aws/final/`;
- polished figures from `outputs/plots/final/`.

The dissertation distinguishes evidence produced across all three baseline
seeds from detailed diagnostics based on representative seed-42 checkpoints.