# Length Generalisation in Sequence Models

This repository contains the code, configurations, experimental results, and report figures for the MSc dissertation project:

**Length Generalisation in Transformer-Based Sequence Models: Failure Taxonomy and Mitigation Strategies**

The project studies whether Transformer sequence models trained on short sequences can generalise to much longer sequences. Instead of only reporting that a model fails, the project analyses **how** it fails, compares positional encoding schemes, tests mitigation strategies, and organises the observed behaviour into a failure taxonomy.

## Project Aim

The central question is:

> When Transformer models are trained on short algorithmic sequences, what kinds of failure behaviour appear when they are evaluated on longer sequences, and can simple mitigation strategies improve length generalisation?

The project focuses on three validated synthetic tasks:

- **Addition**: tests algorithmic reasoning and carry propagation.
- **Delayed copy**: tests long-range memory and sequence reproduction.
- **Reverse**: tests order-sensitive manipulation and position-dependent dependencies.

Associative recall was explored during development, but it is not included in the main comparison because the task was not learned reliably enough to make it a fair extrapolation benchmark.

A small real-world extension using the **SelfRegulationSCP2** time-series dataset is also included to compare synthetic extrapolation behaviour with a noisier applied setting.

## Main Contributions

- A common experimental pipeline for synthetic sequence tasks.
- Baseline Transformer experiments across increasing sequence lengths.
- Comparison of positional encoding methods:
  - learned positional embeddings
  - sinusoidal positional encodings
  - no positional encoding (NoPE)
  - ALiBi
  - RoPE
- Multi-seed robustness analysis using seeds `42`, `123`, and `2024`.
- Error-based plots showing where performance collapses as sequence length increases.
- Attention diagnostics for selected successful-at-train models.
- Mitigation experiments using curriculum, mixed-length training, and randomised padding.
- A proposed failure taxonomy separating different failure behaviours.
- A small real-world extension on SelfRegulationSCP2.

## Repository Structure

```text
configs/                 Experiment configuration files
configs/multiseed/       Multi-seed synthetic experiment configs
configs/realworld/       Real-world time-series experiment configs
data/                    Generated synthetic data and downloaded datasets
experiments/             Experiment notes and supporting files
latex/                   Dissertation LaTeX template and draft material
notebooks/               Exploratory notebooks
outputs/                 Results, checkpoints, plots, and analysis files
scripts/                 Helper scripts for config generation and runs
src/                     Source code for tasks, models, training, and plotting
tests/                   Small validation and debugging scripts
```

## Environment Setup

The project was developed with Python 3.11. A CUDA-enabled GPU is recommended for the full experiment matrix, although smaller runs can be executed locally.

Create and activate an environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On AWS, the same repository can be cloned onto a GPU instance and run from the activated environment. The project uses relative paths, so commands should be run from the repository root.

## Running Synthetic Experiments

Each experiment is controlled by a YAML config file. The usual workflow is:

1. Generate the dataset.
2. Train the model.
3. Evaluate the trained checkpoint.
4. Save CSV results.
5. Generate plots from saved CSVs.

Example for one addition experiment:

```bash
PYTHONPATH=. python -m src.tasks.generate_dataset --config configs/multiseed/addition_learned_seed42.yaml
PYTHONPATH=. python -m src.training.train --config configs/multiseed/addition_learned_seed42.yaml
PYTHONPATH=. python -m src.training.evaluate --config configs/multiseed/addition_learned_seed42.yaml
```

The main synthetic results are stored in:

```text
outputs/results/multiseed_results/
```

These files contain token accuracy and exact-match accuracy for each task, positional encoding, seed, train length, and test length.

## Plotting Final Report Figures

The polished report figures are generated with:

```bash
PYTHONPATH=. python -m src.training.plot_final_report_figures
```

The final report-ready plots are saved in:

```text
outputs/plots/final/
```

Important final figures include:

```text
baseline_training_fit_by_seed.png
addition_eligible_baseline_extrapolation.png
copy_eligible_baseline_extrapolation.png
reverse_eligible_baseline_extrapolation.png
addition_eligible_baseline_error_decomposition.png
copy_eligible_baseline_error_decomposition.png
reverse_eligible_baseline_error_decomposition.png
addition_position_error_heatmap.png
copy_position_error_heatmap.png
reverse_position_error_heatmap.png
addition_learned_baseline_vs_mitigation.png
copy_sinusoidal_baseline_vs_mitigation.png
copy_rope_baseline_vs_mitigation.png
reverse_learned_baseline_vs_mitigation.png
```

The final plotting script also writes summary CSVs used for discussion:

```text
outputs/plots/final/baseline_eligibility_summary.csv
outputs/plots/final/eligible_baseline_error_decomposition.csv
outputs/plots/final/failure_behaviour_summary.csv
outputs/plots/final/mitigation_hypothesis_summary.csv
```

## Attention Diagnostics

Attention analysis is used to inspect whether models that perform well at the training length show changed attention behaviour at longer lengths.

Attention outputs are stored in:

```text
outputs/analysis/attention/
outputs/analysis/attention_summary/
```

The summary plots include normalised entropy, local attention ratio, and average attention distance. These diagnostics support the failure taxonomy by showing that some models lose useful attention structure when evaluated out of distribution.

## Mitigation Experiments

The mitigation experiments test whether exposing models to more varied length conditions improves extrapolation.

The strategies include:

- **Mixed-length training**: training on multiple lengths instead of one fixed short length.
- **Curriculum training**: progressively increasing the training length.
- **Randomised padding**: varying padding structure so the model cannot depend as strongly on fixed absolute positions.

Mitigation results are stored in:

```text
outputs/results/mitigation_results/
outputs/analysis/mitigation_summary/
```

The main interpretation is that these strategies can preserve or recover training-length performance in some cases, but they do not reliably solve extrapolation to much longer sequences. This supports the taxonomy category of **mitigation-resistant extrapolation failure**.

## Real-World Extension

The real-world extension uses the SelfRegulationSCP2 time-series classification dataset.

Example commands:

```bash
PYTHONPATH=. python -m src.training.train_timeseries --config configs/realworld/selfregulationscp2_sinusoidal_seed42.yaml
PYTHONPATH=. python -m src.training.evaluate_timeseries --config configs/realworld/selfregulationscp2_sinusoidal_seed42.yaml
PYTHONPATH=. python -m src.training.plot_realworld_results
```

The real-world results are stored in:

```text
outputs/results/realworld/
outputs/plots/realworld/
```

Unlike the synthetic tasks, the real-world task does not show a sharp exact-match collapse because it is a classification task with noisy input signals. Its role is to provide context rather than replace the controlled synthetic benchmark.

## Failure Taxonomy

The project proposes the following failure categories:

- **Extrapolation collapse**: high performance at the training length followed by near-zero performance at longer lengths.
- **Training-length underfitting**: poor performance even at the training length, meaning the setup is not a valid extrapolation test.
- **Seed instability**: large variation across random seeds.
- **Attention diffusion**: attention becomes less structured or less task-relevant at longer lengths.
- **Mitigation-resistant failure**: mitigation preserves training-length performance but does not improve long-length generalisation.

The taxonomy files are stored in:

```text
outputs/analysis/failure_taxonomy/
```

## Key Findings

- Strong training-length performance does not imply length generalisation.
- Learned and sinusoidal encodings often fit the training length but collapse at longer lengths.
- NoPE and ALiBi frequently underfit on these small synthetic settings, so they should not be interpreted in the same way as models that first solve the training length.
- RoPE can improve some training-length behaviour but does not remove the long-length collapse.
- Multi-seed experiments are necessary because some configurations show unstable results.
- Mitigation experiments show that simple exposure to varied lengths is not enough to force an algorithmic solution.
- Error curves and position-wise diagnostics are more informative than reporting only final accuracy at the longest length.

## Reproducibility Notes

- Run commands from the repository root.
- Use `PYTHONPATH=.` so Python can import modules from `src/`.
- Existing result files may be overwritten when running the same config again.
- GPU runs were performed on AWS for the larger experiment matrix.
- The main random seeds used in the final comparison are `42`, `123`, and `2024`.

## Dissertation Material

The dissertation draft and template files are kept in:

```text
latex/
```

The final report should use the polished figures from `outputs/plots/final/` and the result summaries from `outputs/analysis/` and `outputs/results/`.
