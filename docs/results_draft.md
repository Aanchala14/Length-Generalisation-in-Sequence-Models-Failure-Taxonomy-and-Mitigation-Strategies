# Results Draft

## 4.1 Baseline Length Generalisation

This section reports the baseline Transformer results when trained on short sequences and evaluated on longer sequences. The goal is to test whether the model learns an algorithm that generalises beyond the training length, rather than simply memorising length-specific patterns.

The main observation is that models can often achieve high accuracy at the training length, but exact-match accuracy collapses when evaluated on longer sequences. This pattern is visible across addition, delayed copy, and reverse tasks.

Insert figures:
- outputs/plots/multiseed/addition_exact_degradation_mean_se.png
- outputs/plots/multiseed/copy_exact_degradation_mean_se.png
- outputs/plots/multiseed/reverse_exact_degradation_mean_se.png

Key result:
- Addition fails immediately after train length 16.
- Delayed copy fails when moving from 128 to 256.
- Reverse fails when moving from 128 to 256.

## 4.2 Positional Encoding Comparison

This section compares learned positional embeddings, sinusoidal encodings, NoPE, ALiBi, and RoPE.

The results show that positional encoding affects whether the model can fit the training length, but none of the tested encodings produce reliable extrapolation to length 1024. Learned, sinusoidal, and RoPE can fit some training-length settings, while NoPE and ALiBi often underfit.

Insert tables:
- outputs/plots/multiseed/multiseed_summary.csv
- outputs/plots/multiseed/baseline_comparison.csv

Key result:
- Some positional encodings improve in-distribution performance.
- Exact-match extrapolation remains 0% at the longest test length.

## 4.3 Multi-Seed Robustness

To check robustness, experiments were repeated across three random seeds: 42, 123, and 2024.

The multi-seed results show that some settings are stable, while others have high variance. This means single-seed results are not always reliable.

Insert figure:
- outputs/plots/multiseed/exact_at_train_mean_se.png

Insert table:
- outputs/plots/multiseed/failure_length_table.csv

Key result:
- Copy learned and reverse sinusoidal show seed instability.
- Standard deviation and standard error are necessary for reporting reliability.

## 4.4 Attention-Based Failure Analysis

Attention diagnostics were used to investigate why models fail at longer lengths. The analysis compared attention behaviour at the training length, the first failure length, and length 1024.

The main pattern is that attention becomes more diffuse as sequence length increases. Normalised attention entropy increases, average attention distance grows, and local attention ratio decreases. This suggests that the attention patterns learned at short lengths do not transfer reliably to longer contexts.

Insert figures:
- outputs/analysis/attention_summary/addition_normalised_entropy.png
- outputs/analysis/attention_summary/copy_normalised_entropy.png
- outputs/analysis/attention_summary/reverse_normalised_entropy.png

Insert table:
- outputs/analysis/attention_summary/attention_with_accuracy.csv

Key result:
- Attention behaviour changes substantially at extrapolation lengths.
- This coincides with exact-match accuracy collapsing to 0%.

## 4.5 Mitigation Experiments

Several mitigation strategies were tested, including mixed-length training, curriculum training, and randomised padded mixed-length training.

The results show that these simple mitigation strategies do not solve length extrapolation. Mixed-length training can cause underfitting, while curriculum and randomised padded training preserve training-length performance but still fail at longer lengths.

Insert figure:
- outputs/analysis/mitigation_summary/mitigation_exact_train_vs_1024.png

Insert table:
- outputs/analysis/mitigation_summary/mitigation_summary.csv

Key result:
- Curriculum and randomised training keep Exact@Train high.
- Exact@256, Exact@512, and Exact@1024 remain 0%.

## 4.6 Failure Taxonomy

Based on the experimental results, five failure types were identified.

Insert table:
- outputs/analysis/failure_taxonomy/failure_taxonomy_compact.csv

The taxonomy summarises the main failure modes observed in the project: extrapolation collapse, training-length underfitting, seed instability, attention diffusion, and mitigation-resistant failure.

Overall, the results support the claim that the tested Transformer models learn length-specific solutions rather than robust length-general algorithms.