# Zero-shot benchmarks (lm-evaluation-harness)

Accuracy in %, ± one standard error. acc_norm where the task has it (length-normalised by bytes), otherwise acc. Precision: float32.

| Model | hellaswag | piqa | arc_easy | arc_challenge | winogrande | sciq | openbookqa | lambada_openai | Average |
|---|---|---|---|---|---|---|---|---|---|
| Ours (110M) | 29.5 ± 0.5 | 60.1 ± 1.1 | 43.1 ± 1.0 | 24.1 ± 1.2 | 51.7 ± 1.4 | 65.4 ± 1.5 | 28.0 ± 2.0 | 22.8 ± 0.6 | 40.6 |
| GPT-2 small (124M) | 31.1 ± 0.5 | 62.6 ± 1.1 | 39.5 ± 1.0 | 22.7 ± 1.2 | 51.8 ± 1.4 | 64.4 ± 1.5 | 27.2 ± 2.0 | 32.6 ± 0.7 | 41.5 |
| Ours (336M) | 35.4 ± 0.5 | 63.3 ± 1.1 | 48.3 ± 1.0 | 28.0 ± 1.3 | 51.6 ± 1.4 | 70.4 ± 1.4 | 31.8 ± 2.1 | 30.7 ± 0.6 | 44.9 |
| GPT-2 medium (355M) | 39.4 ± 0.5 | 66.4 ± 1.1 | 43.6 ± 1.0 | 25.1 ± 1.3 | 53.2 ± 1.4 | 67.2 ± 1.5 | 30.2 ± 2.1 | 43.1 ± 0.7 | 46.0 |
| GPT-2 large (774M) | 45.3 ± 0.5 | 69.2 ± 1.1 | 46.6 ± 1.0 | 25.1 ± 1.3 | 55.5 ± 1.4 | 69.3 ± 1.5 | 31.2 ± 2.1 | 47.7 ± 0.7 | 48.7 |

Chance: hellaswag 25%, piqa 50%, arc_easy 25%, arc_challenge 25%, winogrande 50%, sciq 25%, openbookqa 25%, lambada_openai 0%.

## Check: our wrapper vs the harness's own Hugging Face model (GPT-2 small)

| Task | Metric | Our wrapper | Harness HF | Difference |
|---|---|---|---|---|
| hellaswag | acc | 28.94 | 28.93 | +0.01 |
| hellaswag | acc_norm | 31.12 | 31.12 | +0.00 |
| piqa | acc | 62.89 | 62.89 | +0.00 |
| piqa | acc_norm | 62.57 | 62.57 | +0.00 |
| arc_easy | acc | 43.77 | 43.81 | -0.04 |
| arc_easy | acc_norm | 39.52 | 39.52 | +0.00 |
| arc_challenge | acc | 18.94 | 18.94 | +0.00 |
| arc_challenge | acc_norm | 22.70 | 22.70 | +0.00 |
| winogrande | acc | 51.78 | 51.78 | +0.00 |
| sciq | acc | 75.30 | 75.20 | +0.10 |
| sciq | acc_norm | 64.40 | 64.30 | +0.10 |
| openbookqa | acc | 16.40 | 16.40 | +0.00 |
| openbookqa | acc_norm | 27.20 | 27.20 | +0.00 |
| lambada_openai | acc | 32.56 | 32.56 | +0.00 |
