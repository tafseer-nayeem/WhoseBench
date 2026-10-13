# Scorer sensitivity to valid answer variants

This directory contains item-level records and summaries for paired, meaning-preserving spelling, date, and time variants. The **benchmark-specific** DROP and TruthfulQA measures should be read separately. These are controlled scorer-side comparisons, **not changes in model accuracy**.

| Benchmark | Measure | Original | Variant | Change | 95% CI for change |
| --- | --- | ---: | ---: | ---: | ---: |
| DROP | Exact match | 100.0% | 33.3% | -66.7 pp | [-92.3, -36.4] pp |
| DROP | Token F1 | 100.0% | 77.2% | -22.8 pp | [-41.6, -5.2] pp |
| TruthfulQA | BLEU correctness | 97.0% | 97.0% | 0.0 pp | [0.0, 0.0] pp |
| TruthfulQA | ROUGE-1 correctness | 97.0% | 97.0% | 0.0 pp | [0.0, 0.0] pp |
| TruthfulQA | ROUGE-2 correctness | 100.0% | 93.9% | -6.1 pp | [-16.1, 0.0] pp |
| TruthfulQA | ROUGE-1 similarity | 100.0 | 88.9 | -11.1 points | [-13.5, -9.0] points |

The DROP slice contains **12 eligible pairs from 11 questions**. The clean TruthfulQA slice contains **33 pairs from 25 questions**. Confidence intervals use question-clustered bootstrap resampling. The first two rows come from [`paired_benchmark_specific_summary.csv`](paired_benchmark_specific_summary.csv); the TruthfulQA generation rows come from [`truthfulqa_generation_similarity_summary.csv`](truthfulqa_generation_similarity_summary.csv). Corresponding item-level records are [`official_drop_paired_records.csv`](official_drop_paired_records.csv) and [`truthfulqa_generation_similarity_records.csv`](truthfulqa_generation_similarity_records.csv).

The other `exact_match_*`, `outcome_*`, and `scorer_control_*` files document broader historical/proxy sensitivity analyses. Do not combine their scores with the benchmark-specific table above into a single accuracy claim. Reproduction commands are in the [root README](../../README.md); the scripts are in [`scripts/sensitivity/`](../../scripts/sensitivity/).
