# Reading Guide for Supervisor Update

## Purpose

This guide explains how the current project materials fit together. The aim is to make the update easy to review without requiring the supervisor to inspect every raw CSV or implementation file.

## Project Question

The project investigates whether small Transformer sequence models learn length-general algorithms or whether they learn solutions that work only at the training length.

The experimental setup follows a train-short/test-long design:

- Train on short or moderate sequence lengths.
- Test on longer unseen lengths up to 1024.
- Compare positional encodings and mitigation strategies.
- Diagnose failures using accuracy, robustness, and attention behaviour.

## Why These Synthetic Tasks Were Chosen

The project uses synthetic algorithmic tasks because they allow controlled testing of length generalisation.

- Addition tests arithmetic carry propagation and symbolic computation.
- Delayed copy tests long-range memory after a separator token.
- Reverse tests precise positional remapping.
- Associative recall was explored as an additional memory task, but the main completed matrix focuses on addition, delayed copy, and reverse.

The original simple copy task was replaced with delayed copy because simple copy was too easy and did not meaningfully test long-range memory.

## What Has Been Completed

The current completed experimental package includes:

- Baseline Transformer experiments.
- Positional encoding comparison: learned, sinusoidal, NoPE, ALiBi, and RoPE.
- Multi-seed robustness across seeds 42, 123, and 2024.
- Attention diagnostics at train length, first failure length, and length 1024.
- Mitigation experiments using mixed-length, curriculum, and randomised padded training.
- Failure taxonomy summarising the observed failure modes.

## Suggested Reading Order

### 1. Start With the Project Design

Read the project plan first to check whether the task design and experiment structure answer the research question.

Relevant file:

- `Project_time_plan.docx`

Key thing to check:

- Are the selected synthetic tasks sufficient for testing length generalisation?
- Is the train-short/test-long design appropriate?

### 2. Review the Main Update Deck

Read:

- `week9_project_update.pptx`

This deck gives the shortest high-level summary of the project state. It explains what has changed since the previous update, what experiments are complete, and what the current interpretation is.

Key thing to check:

- Is the story clear enough for the dissertation?
- Are the conclusions supported by the reported evidence?

### 3. Inspect the Multi-Seed Results

Main files:

- `outputs/plots/multiseed/multiseed_summary.csv`
- `outputs/plots/multiseed/failure_length_table.csv`
- `outputs/plots/multiseed/baseline_comparison.csv`

Main figures:

- `outputs/plots/multiseed/addition_exact_degradation_mean_se.png`
- `outputs/plots/multiseed/copy_exact_degradation_mean_se.png`
- `outputs/plots/multiseed/reverse_exact_degradation_mean_se.png`
- `outputs/plots/multiseed/exact_at_train_mean_se.png`
- `outputs/plots/multiseed/generalisation_gap_mean.png`

Key finding:

- Models often fit the training length but exact-match accuracy collapses at extrapolation lengths.
- Addition fails immediately after length 16.
- Delayed copy and reverse fail when moving from length 128 to 256.

### 4. Inspect Attention Diagnostics

Main files:

- `outputs/analysis/attention_summary/attention_with_accuracy.csv`
- `outputs/analysis/attention_summary/attention_overall.csv`

Main figures:

- `outputs/analysis/attention_summary/addition_normalised_entropy.png`
- `outputs/analysis/attention_summary/copy_normalised_entropy.png`
- `outputs/analysis/attention_summary/reverse_normalised_entropy.png`

Key finding:

- Attention behaviour changes substantially at longer lengths.
- Normalised attention entropy increases.
- Local attention ratio decreases.
- These changes coincide with exact-match accuracy falling to 0%.

### 5. Review Mitigation Results

Main file:

- `outputs/analysis/mitigation_summary/mitigation_summary.csv`

Main figure:

- `outputs/analysis/mitigation_summary/mitigation_exact_train_vs_1024.png`

Key finding:

- Mixed-length training can cause underfitting.
- Curriculum and randomised padded training preserve training-length accuracy.
- None of the tested mitigation strategies improve exact-match extrapolation to 256, 512, or 1024.

### 6. Review the Failure Taxonomy

Main files:

- `outputs/analysis/failure_taxonomy/failure_taxonomy.csv`
- `outputs/analysis/failure_taxonomy/failure_taxonomy_compact.csv`

The taxonomy identifies five failure modes:

- Extrapolation collapse.
- Training-length underfitting.
- Seed instability.
- Attention diffusion.
- Mitigation-resistant failure.

Key thing to check:

- Are these categories clearly separated?
- Should any category be renamed or merged before dissertation writing?

## Current Interpretation

The current evidence suggests that the tested Transformer models learn length-specific solutions. Some models solve the training length almost perfectly, but the learned computation does not transfer to longer sequences.

Simple mitigation strategies are not enough to force algorithmic length generalisation. This strengthens the project argument because the failure persists across tasks, positional encodings, seeds, attention diagnostics, and mitigation attempts.

## Questions for Supervisor Feedback

1. Are the failure taxonomy categories convincing?
2. Should ALiBi and NoPE be discussed mainly as underfitting cases rather than extrapolation cases?
3. Is one small final experiment needed, or is the current experimental package sufficient?
4. Is the mitigation interpretation fair: simple exposure to multiple shorter lengths does not produce length-general algorithms?
5. Should the dissertation prioritise attention analysis or mitigation results in the discussion?

## Recommended Next Work

The main experimental work is complete through Week 9. The next priority should be writing:

- Results section.
- Discussion section.
- Failure taxonomy explanation.
- Link back to the literature on positional encodings and train-short/test-long extrapolation.

Week 10 can be kept as a small improvement window if supervisor feedback suggests one additional experiment.
