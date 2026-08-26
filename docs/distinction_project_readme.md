# Distinction-Level Project and Report Plan

This document records the main steps needed to turn the current project into a strong distinction-level dissertation and presentation. The experimental work is already substantial; the main priority now is to write, structure, validate, and present it in a rigorous research style.

## Current Project Strength

The project already has a coherent research arc:

1. Controlled synthetic benchmark.
2. Train-short/test-long evaluation protocol.
3. Positional encoding comparison.
4. Multiple random seeds.
5. Performance degradation curves.
6. Attention-based failure diagnostics.
7. Mitigation experiments.
8. Failure taxonomy.
9. Small real-world extension using SelfRegulationSCP2.

This is a strong experimental package. The main risk is not lack of work, but unclear writing, weak justification, or overclaiming the results.

## Main Goal for a High Mark

The dissertation should show that the project is not simply a collection of experiments. It should read as a controlled investigation into how Transformer length generalisation fails, supported by systematic evidence.

The report should repeatedly connect back to the central research question:

> When a Transformer model is trained only on short sequences, what fails when it is evaluated on longer sequences, and can these failures be systematically classified?

## Highest-Impact Improvements

### 1. Make the Failure Taxonomy Formal

The taxonomy should be presented as one of the main contributions of the project.

For each failure type, include:

- Failure type.
- Operational criterion.
- Evidence used.
- Tasks affected.
- Interpretation.
- Limitation.

Example:

```text
Extrapolation collapse:
A model achieves high exact-match accuracy at the training length but falls below 10% exact-match accuracy at the first longer test length.
```

The taxonomy should not look like an informal summary. It should look like a framework derived from the experiments.

### 2. Treat Mitigation as Analysis, Not Just Failed Experiments

The title includes mitigation, so the mitigation section must be written carefully.

For each mitigation strategy, use this structure:

```text
Hypothesis:
What was the method expected to improve?

Protocol:
How was it tested?

Result:
What happened?

Interpretation:
Why might it have failed?

Conclusion:
What does this reveal about length generalisation?
```

Important interpretation:

- Mixed-length training can underfit.
- Curriculum and randomised padded training can preserve training-length performance.
- However, these methods still fail at longer lengths.
- Therefore, simple exposure to multiple shorter lengths is not sufficient to force algorithmic length generalisation.

This makes the mitigation result valuable even when the mitigation does not succeed.

### 3. Include a Reproducibility Section

A strong dissertation should make the examiner feel that the work can be reproduced.

Include:

- Code structure.
- Dataset generation procedure.
- YAML configuration system.
- Train/test lengths.
- Seeds: 42, 123, 2024.
- Hardware setup: local machine and AWS GPU instance.
- Output structure.
- Plot generation scripts.

Useful files to reference:

```text
src/tasks/
src/models/
src/training/
configs/multiseed/
configs/mitigation/
configs/realworld/
outputs/results/
outputs/plots/
outputs/analysis/
```

### 4. Add Implementation Validation

The professor specifically raised concern about NoPE and ALiBi. The report should explicitly say how the project avoids misinterpreting poor results.

Key point:

> A configuration is only treated as a length-generalisation failure if it first learns the training length. If it fails at the training length, it is classified as training-length underfitting.

Use the control results to support this:

- Addition learned and sinusoidal fit the training length.
- Delayed copy sinusoidal and RoPE fit the training length.
- Reverse learned fits the training length.
- NoPE and ALiBi often underfit at the training length, so their results are interpreted cautiously.

This is important for academic credibility.

### 5. Strengthen the Literature Connection

The report should not only describe experiments. It should compare the findings with the papers.

Important connections:

- Vaswani et al. introduced Transformers and sinusoidal positional encoding.
- Press et al. proposed ALiBi for train-short/test-long language modelling.
- Kazemnejad et al. studied positional encoding and length generalisation on reasoning tasks, including NoPE.
- RoPE provides a relative/rotary positional mechanism.
- Rough Transformers motivates the real-world long time-series extension.

Critical comparison:

> Prior work often reports that positional encodings can improve length extrapolation, but the current experiments show that this depends strongly on task type and whether the model first learns the training length. In this project, some positional encodings do not fail by extrapolating poorly; they fail by underfitting the original task.

This kind of critical discussion helps push the report toward distinction level.

## Optional Extra Experiment

No major new experiment is necessary.

If there is time for one small addition, add a simple baseline:

```text
Synthetic tasks:
Random/chance baseline for token accuracy and exact-match accuracy.

Real-world task:
Majority-class baseline for SelfRegulationSCP2.
```

This would help contextualise the real-world accuracy around 49-54%.

This is optional. The main gain now is writing quality.

## Results Writing Strategy

Use this pattern throughout:

```text
Question -> Setup -> Figure/Table -> Observation -> Interpretation -> Limitation
```

Example:

```text
This experiment asks whether a Transformer trained at one sequence length learns a rule that transfers to longer lengths. Figure X shows exact-match accuracy as test length increases. Although some models achieve near-perfect accuracy at the training length, exact-match accuracy falls to 0% at the first longer test length. This indicates that the model has learned a length-specific solution rather than a length-general algorithm.
```

## Chapter Priorities

Recommended writing order:

1. Chapter 3: Methodology and Experimental Design.
2. Chapter 4: Synthetic Benchmark Results.
3. Chapter 5: Failure Analysis and Mitigation.
4. Chapter 6: Real-World Extension.
5. Chapter 2: Related Work.
6. Chapter 1: Introduction.
7. Discussion, Conclusion, Abstract.

This order is practical because the experiments are already complete.

## Figure and Table Checklist

Core figures:

```text
outputs/plots/multiseed/addition_exact_degradation_mean_se.png
outputs/plots/multiseed/copy_exact_degradation_mean_se.png
outputs/plots/multiseed/reverse_exact_degradation_mean_se.png
outputs/plots/multiseed/exact_at_train_mean_se.png
outputs/plots/multiseed/generalisation_gap_mean.png
outputs/analysis/attention_summary/addition_normalised_entropy.png
outputs/analysis/attention_summary/copy_normalised_entropy.png
outputs/analysis/attention_summary/reverse_normalised_entropy.png
outputs/analysis/mitigation_summary/mitigation_exact_train_vs_1024.png
outputs/plots/realworld/selfregulationscp2_accuracy_mean_se.png
```

Core tables:

```text
outputs/plots/multiseed/multiseed_summary.csv
outputs/plots/multiseed/failure_length_table.csv
outputs/plots/multiseed/baseline_comparison.csv
outputs/analysis/attention_summary/attention_with_accuracy.csv
outputs/analysis/mitigation_summary/mitigation_summary.csv
outputs/analysis/failure_taxonomy/failure_taxonomy_compact.csv
outputs/plots/realworld/selfregulationscp2_summary.csv
```

Every figure should have:

- A clear caption.
- A direct interpretation in the paragraph after the figure.
- A connection to the research question.

## Presentation Strategy

The presentation should not show every result. It should show the research story:

1. Problem: Transformers fail to generalise to longer sequences.
2. Method: controlled train-short/test-long benchmark.
3. Evidence: degradation curves.
4. Robustness: multiple seeds and positional encodings.
5. Diagnosis: attention behaviour changes at longer lengths.
6. Mitigation: simple strategies do not solve the problem.
7. Contribution: failure taxonomy.
8. Extension: real-world task behaves differently from synthetic exact tasks.
9. Conclusion: failures are task-dependent and must be classified, not only measured.

## Claims to Make Carefully

Strong claims that are safe:

- The tested models often fit the training length but fail at longer lengths.
- Exact-match accuracy is a strict and appropriate metric for synthetic algorithmic tasks.
- NoPE and ALiBi often underfit in this setup, so they should be interpreted cautiously.
- Attention statistics change as sequence length increases.
- Simple mitigation strategies were insufficient in this experimental setting.
- Real-world classification does not show the same catastrophic collapse as exact synthetic sequence prediction.

Claims to avoid:

- Do not claim all Transformers fail at length generalisation.
- Do not claim ALiBi or NoPE are generally bad.
- Do not claim the mitigation strategies can never work.
- Do not claim the real-world extension is state-of-the-art.
- Do not claim attention diagnostics fully explain the failures.

## Final Distinction-Level Checklist

Before submission, ensure:

- The LaTeX template compiles.
- `dissertation(1).cls` is renamed to `dissertation.cls` in Overleaf.
- The bibliography is replaced with real project references.
- All figures are readable in the PDF.
- All tables fit the page.
- The methodology explains why each experiment was done.
- The results section interprets every figure/table.
- The discussion connects results back to the literature.
- The taxonomy is presented as a contribution.
- Limitations are explicit and mature.
- The conclusion clearly states what was learned.

## Overall Assessment

The experimental work is already strong. The remaining path to a very high mark is mostly:

```text
70% writing and argument
15% reproducibility and cleanup
10% figures and tables
5% optional small baseline
```

The project should be written as:

> A controlled study showing that length generalisation failure is not a single phenomenon, but a collection of distinguishable failure modes depending on task, positional encoding, optimisation stability, attention behaviour, and mitigation response.

