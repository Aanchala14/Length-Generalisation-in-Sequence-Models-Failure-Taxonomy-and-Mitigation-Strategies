# Supervisor Meeting Explanation Guide

Meeting date: 25 August 2026

Project title: Length Generalisation in Transformer-Based Sequence Models: Failure Taxonomy and Mitigation Strategies

This document is a preparation guide for explaining the current project progress to the supervisor. It summarises what was done, why it was done, what the results show, and how the findings should be interpreted.

## 1. Short Opening Explanation

The project investigates whether Transformer models trained on short sequences learn rules that generalise to longer sequences, or whether they learn solutions that only work at the training length.

The main experimental design is a train-short/test-long setup. I trained small Transformer models on controlled synthetic sequence tasks and then evaluated them on progressively longer unseen lengths. I compared five positional encoding methods, repeated experiments across three seeds, analysed degradation curves and attention behaviour, tested mitigation strategies, and added a small real-world extension using SelfRegulationSCP2.

The main finding is that high training-length accuracy does not imply length generalisation. Some models solve the training length almost perfectly but collapse to 0% exact-match accuracy at longer lengths. The project therefore focuses not only on reporting accuracy, but on classifying different failure modes.

## 2. What Changed Since the Last Update

Since the latest progress presentation, the project has moved from baseline synthetic experiments to a more complete experimental package.

Completed work:

- Re-ran the main synthetic experiments after fixing the positional encoding implementation issue.
- Completed the full multi-seed matrix:
  - 3 tasks: addition, delayed copy, reverse.
  - 5 positional encodings: learned, sinusoidal, NoPE, ALiBi, RoPE.
  - 3 seeds: 42, 123, 2024.
  - Total: 45 main synthetic runs.
- Produced degradation curves showing exact-match and token accuracy across sequence lengths.
- Produced mean, standard deviation, and standard error summaries.
- Added attention diagnostics:
  - normalised attention entropy,
  - average attention distance,
  - local attention ratio.
- Ran mitigation experiments:
  - mixed-length training,
  - curriculum training,
  - randomised padded training.
- Built a failure taxonomy from the results.
- Added a small real-world extension using SelfRegulationSCP2.
- Started writing the dissertation in the university LaTeX template.

## 3. How to Explain the Plots and Curves

The plots are clear enough for the meeting and for the report, as long as each one is explained with the correct message. The important point is not only what the curve looks like, but what inference should be drawn from it.

When presenting the plots, use this structure:

1. Say what the plot measures.
2. Point out the training length.
3. Point out what happens at the first longer test length.
4. Explain whether this is extrapolation collapse, underfitting, seed instability, or a real-world contrast.

### Addition Exact-Match Degradation Curve

What the plot shows:

- Exact-match accuracy on addition as digit length increases.
- Results are averaged across three seeds.
- Error bars show standard error.

Main inference:

> Learned and sinusoidal positional encodings solve the training length reasonably well, but both collapse immediately when the digit length increases from 16 to 32.

How to explain it:

The model can learn addition at the training length, but it does not learn a length-general arithmetic algorithm. The failure happens at the first extrapolation length, so the degradation is not gradual. This is a clear example of extrapolation collapse.

Important numbers:

- Learned Exact@Train: 92.0%.
- Sinusoidal Exact@Train: 93.7%.
- Both Exact@Longest: 0.0%.
- First failure length: 32.

### Delayed Copy Exact-Match Degradation Curve

What the plot shows:

- Exact-match accuracy on delayed copy as sequence length increases.
- Training length is 128.
- Test lengths are 128, 256, 512, and 1024.

Main inference:

> Sinusoidal and RoPE fit the training length almost perfectly, but fail completely when the length doubles to 256.

How to explain it:

This shows that the model can remember and reproduce sequences at the training length, but the learned memory solution is tied to that length. It does not transfer when the delay and sequence length increase.

Important numbers:

- Sinusoidal Exact@Train: 99.93%.
- RoPE Exact@Train: 100.0%.
- Both Exact@1024: 0.0%.
- First failure length: 256.

### Reverse Exact-Match Degradation Curve

What the plot shows:

- Exact-match accuracy on reverse as sequence length increases.
- Training length is 128.

Main inference:

> Learned positional embeddings solve reverse perfectly at the training length but fail immediately at length 256.

How to explain it:

Reverse requires precise positional remapping. The result suggests that the model learns a position-specific mapping for length 128 rather than a general reversal rule.

Important numbers:

- Learned Exact@Train: 100.0%.
- Learned Exact@1024: 0.0%.
- First failure length: 256.

### Exact-at-Train Plot

What the plot shows:

- How well each positional encoding learns the training length.
- This helps distinguish genuine extrapolation failure from underfitting.

Main inference:

> Some models fail because they cannot extrapolate, but others fail because they never learn the training task.

How to explain it:

This plot is important for interpreting NoPE and ALiBi. If a model has near-zero exact-match accuracy at the training length, then its long-length failure should not be called extrapolation collapse. It should be classified as training-length underfitting.

### Generalisation Gap Plot

What the plot shows:

- The difference between Exact@Train and Exact@Longest.

Main inference:

> The largest gaps occur when models fit the training length but completely fail at the longest length.

How to explain it:

A high generalisation gap means the model looks successful under the training distribution but fails when length changes. This is the central evidence that high training-length accuracy does not imply length-general behaviour.

### Attention Entropy Plots

What the plots show:

- Normalised attention entropy at training length, first failure length, and longest length.

Main inference:

> Attention becomes more diffuse as the sequence length increases, and this coincides with exact-match collapse.

How to explain it:

For addition and reverse, the model needs precise attention patterns to align positions or propagate information. As length increases, attention becomes more spread out. This suggests that the short-length attention pattern does not transfer cleanly to longer contexts.

Important caution:

> This is diagnostic evidence, not proof of causality.

### Mitigation Plot

What the plot shows:

- Exact-match accuracy at the training length compared with exact-match accuracy at length 1024 for mitigation experiments.

Main inference:

> Some mitigation strategies preserve training-length performance, but they still do not improve long-length exact-match accuracy.

How to explain it:

Curriculum and randomised training do not simply break the model. They can keep training-length accuracy high. However, they still fail at length 1024. This supports the mitigation-resistant failure category.

### Real-World Extension Plot

What the plot shows:

- Classification accuracy on SelfRegulationSCP2 as input length increases from 256 to 1152.

Main inference:

> The real-world classification task does not show catastrophic collapse. Accuracy stays roughly around 49-54%.

How to explain it:

This is different from the synthetic exact-match tasks because SelfRegulationSCP2 is a classification task. The model only needs one final label, and useful local or partial features may remain available at longer lengths. This supports the conclusion that length-generalisation failure is task-dependent.

Important wording:

> The real-world extension is a contrast and validation check, not a state-of-the-art comparison.

## 4. How the Professor's Feedback Was Addressed

### Feedback: Complete multiple-seed experiments and report mean/std.

Done.

The main matrix now uses seeds 42, 123, and 2024. The summaries report mean, standard deviation, and standard error.

### Feedback: Show degradation curves, not only train length and longest length.

Done.

The plots now show how performance changes across all evaluated lengths. This makes it possible to see where failure begins.

### Feedback: Check NoPE and ALiBi because results looked suspicious.

Done carefully.

The positional encoding implementation was checked and corrected. NoPE and ALiBi are now included in the full matrix. However, their results are interpreted cautiously because they often fail to learn the training length. Therefore, they are classified as training-length underfitting rather than pure extrapolation failure.

### Feedback: Add baselines and compare against them.

Done.

The benchmark includes learned positional embeddings as a baseline, plus sinusoidal, NoPE, ALiBi, and RoPE. There are also baseline comparisons and mitigation controls.

### Feedback: Analyse failure patterns, not only final accuracy.

Done.

The project now includes:

- failure length analysis,
- degradation curves,
- multi-seed robustness,
- attention diagnostics,
- mitigation response,
- failure taxonomy.

### Feedback: Leave associative recall out for now.

Done.

Associative recall is not part of the main positional encoding comparison.

## 5. Main Synthetic Results

The strongest pattern is:

> Some models fit the training length very well, but exact-match accuracy collapses immediately at longer lengths.

Important examples:

- Addition with learned positional embeddings:
  - Exact@Train: 92.0%.
  - Exact@Longest: 0.0%.
  - First failure length: 32.

- Addition with sinusoidal positional encoding:
  - Exact@Train: 93.7%.
  - Exact@Longest: 0.0%.
  - First failure length: 32.

- Delayed copy with sinusoidal positional encoding:
  - Exact@Train: 99.93%.
  - Exact@Longest: 0.0%.
  - First failure length: 256.

- Delayed copy with RoPE:
  - Exact@Train: 100.0%.
  - Exact@Longest: 0.0%.
  - First failure length: 256.

- Reverse with learned positional embeddings:
  - Exact@Train: 100.0%.
  - Exact@Longest: 0.0%.
  - First failure length: 256.

Interpretation:

The model can learn a solution for the training length, but the learned solution is not length-general. It behaves more like a length-specific mapping than a general algorithm.

## 6. Positional Encoding Interpretation

The positional encoding comparison should be explained carefully.

The results do not show that one positional encoding universally solves length generalisation. Instead:

- Learned and sinusoidal encodings can fit addition at the training length but fail to extrapolate.
- Sinusoidal and RoPE can fit delayed copy at the training length but fail to extrapolate.
- Learned positional embeddings can fit reverse at the training length but fail to extrapolate.
- NoPE and ALiBi often do not fit the training length in this setup.

Important wording:

> NoPE and ALiBi are not simply described as extrapolation failures. In this project, they often fail at the training length, so they are better classified as training-length underfitting cases.

This distinction is important because it shows that the project is not overclaiming.

## 7. Multi-Seed Robustness

The multi-seed results show that some configurations are stable, but others are seed-sensitive.

Examples:

- Reverse with learned positional embeddings is stable at the training length.
- Delayed copy with learned positional embeddings has high variation across seeds.
- Reverse with sinusoidal positional encoding also shows high seed variation.

Interpretation:

Single-seed results can be misleading. Reporting mean, standard deviation, and standard error makes the conclusions more robust.

This supports one category in the taxonomy: seed instability.

## 8. Attention Diagnostics

The attention analysis asks:

> What changes inside the model when sequence length increases?

Metrics used:

- Normalised attention entropy.
- Average attention distance.
- Local attention ratio.

Main pattern:

- Attention becomes more diffuse at longer lengths.
- Normalised attention entropy increases.
- Local attention structure weakens.
- This coincides with exact-match accuracy collapse.

Example:

For addition with learned positional embeddings:

- Length 16:
  - exact match: 94.1%.
  - normalised entropy: 0.16.

- Length 32:
  - exact match: 0%.
  - normalised entropy: 0.28.

- Length 1024:
  - exact match: 0%.
  - normalised entropy: 0.59.

Interpretation:

The attention pattern that works at the training length does not transfer cleanly to longer contexts. However, this should be described as diagnostic evidence, not full proof of causality.

Good wording:

> Attention diagnostics suggest that model behaviour changes systematically at extrapolation lengths, but further mechanistic experiments would be needed to prove causality.

## 9. Mitigation Experiments

Mitigation strategies tested:

- mixed-length training,
- curriculum training,
- randomised padded training.

The goal was to test whether exposing the model to more varied lengths would improve extrapolation.

Main findings:

- Mixed-length training sometimes underfits.
- Curriculum training can preserve training-length accuracy.
- Randomised padded training can preserve training-length accuracy.
- However, longer-length exact-match accuracy remains 0% in the tested cases.

Important examples:

- Copy curriculum sinusoidal:
  - Exact@Train: 100%.
  - Exact@1024: 0%.

- Copy curriculum RoPE:
  - Exact@Train: 100%.
  - Exact@1024: 0%.

- Reverse curriculum learned:
  - Exact@Train: 100%.
  - Exact@1024: 0%.

Interpretation:

Simple exposure to more lengths is not sufficient to force a length-general algorithm. The mitigation failures are still useful because they show the failure is not just caused by training on a single fixed length.

## 10. Failure Taxonomy

The taxonomy is the main interpretive contribution of the project.

It separates failures into five categories:

### 1. Extrapolation Collapse

The model fits the training length but fails at longer lengths.

Example:

- reverse learned,
- addition learned,
- delayed copy sinusoidal.

### 2. Training-Length Underfitting

The model fails even at the training length.

Example:

- many NoPE and ALiBi configurations.

### 3. Seed Instability

The model's success depends strongly on random seed.

Example:

- delayed copy learned,
- reverse sinusoidal.

### 4. Attention Diffusion

Attention becomes more diffuse as sequence length increases.

Evidence:

- increased normalised attention entropy,
- reduced local attention ratio,
- higher average attention distance.

### 5. Mitigation-Resistant Failure

Mitigation preserves training-length performance but does not improve extrapolation.

Example:

- copy curriculum sinusoidal,
- copy curriculum RoPE,
- reverse curriculum learned.

Why the taxonomy matters:

It prevents different failures from being treated as the same thing. A model that cannot learn the training length is not failing in the same way as a model that learns the training length and then collapses beyond it.

## 11. Real-World Extension

Dataset:

- SelfRegulationSCP2 from the UEA time-series archive.

Protocol:

- train on prefix length 256,
- test on 256, 512, 1024, and 1152,
- use the same five positional encodings,
- repeat across three seeds.

Main result:

Accuracy remains relatively stable across length, mostly around 49-54%.

At full length 1152:

- learned: 53.52%,
- sinusoidal: 53.52%,
- ALiBi: 51.48%,
- NoPE: 50.37%,
- RoPE: 49.07%.

Interpretation:

The real-world task does not show catastrophic collapse like the synthetic exact-match tasks.

Reason:

SelfRegulationSCP2 is a classification task. The model only needs to predict one class label, and useful partial or local features may remain available at longer lengths. In contrast, synthetic tasks require exact structured sequence outputs, where one wrong token causes exact-match failure.

Good wording:

> The real-world extension suggests that length generalisation failure is task-dependent. It is most severe in exact algorithmic sequence-generation tasks, while real-world classification may show flatter performance because partial evidence can still support the final prediction.

## 12. Main Inference from the Whole Project

The main inference is:

> Length generalisation failure in Transformers is not one single phenomenon. It depends on task type, positional encoding, training stability, attention behaviour, and response to mitigation.

The project's contribution is not just that models fail. The contribution is that the failures are separated and interpreted systematically.

The strongest final message:

> High training-length accuracy does not prove that a Transformer has learned a length-general algorithm.

## 13. How to Explain the Project in the Meeting

Suggested speaking flow:

1. Start with the research question.
2. Explain the train-short/test-long design.
3. Explain the three synthetic tasks.
4. Show the degradation curves.
5. Explain that some models fit train length but fail immediately at longer lengths.
6. Explain why NoPE and ALiBi are treated as underfitting in this setup.
7. Explain multi-seed robustness.
8. Explain attention diagnostics.
9. Explain mitigation results.
10. Present the taxonomy as the main contribution.
11. Explain the real-world extension as a contrast, not the main result.
12. Ask for feedback on framing and report structure.

## 14. Questions to Ask the Supervisor

Useful questions:

1. Is the failure taxonomy framed clearly enough as the main contribution?
2. Is it acceptable to classify NoPE and ALiBi mainly as training-length underfitting in this setup?
3. Should the mitigation section emphasise negative results as evidence, or should it be shortened?
4. Is the real-world extension useful as a contrast, or should it stay as a small optional section?
5. Should the dissertation prioritise taxonomy and failure analysis over the raw positional encoding comparison?
6. Are the current chapter divisions appropriate?
7. Is there any small experiment that would strengthen the final report, or should the remaining time focus on writing?

## 15. Claims to Make Carefully

Safe claims:

- The tested models often fit the training length but fail at longer lengths.
- The failure is clearest under exact-match sequence evaluation.
- Positional encoding affects training-length learning but does not guarantee extrapolation.
- NoPE and ALiBi often underfit in this setup.
- Attention diagnostics suggest attention behaviour changes at longer lengths.
- Simple mitigation strategies were insufficient in the tested setting.
- The real-world classification task behaves differently from synthetic exact tasks.

Claims to avoid:

- Do not claim all Transformers fail at length generalisation.
- Do not claim ALiBi or NoPE are generally bad.
- Do not claim RoPE cannot help in general.
- Do not claim the attention diagnostics prove causality.
- Do not claim the real-world extension is state-of-the-art.
- Do not claim mitigation strategies can never work.

## 16. Final Meeting Summary

The project is now at the stage where the main experiments are complete and the focus has shifted to writing and interpretation.

The report should be written around this core story:

> I built a controlled benchmark for Transformer length generalisation, showed that several models learn training-length solutions that collapse at longer lengths, analysed the failures using degradation curves and attention diagnostics, tested simple mitigation strategies, and organised the findings into a failure taxonomy. A small real-world extension showed that these failures are most severe in exact algorithmic sequence tasks and less catastrophic in classification.

This is the story to communicate clearly in the meeting.
