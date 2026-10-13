# Analysis scripts

Run commands from the repository root. [`validate_outputs.py`](validate_outputs.py) checks selected released outputs using only Python's standard library:

```bash
python3 scripts/validate_outputs.py
```

| Directory | Role | Typical entry points |
| --- | --- | --- |
| [`convention/`](convention/) | Acquire text benchmarks, scan convention signals, aggregate variation tables. | `download_benchmark_bundle.py`, `scan_benchmark_variations.py` |
| [`geographical_profiling/`](geographical_profiling/) | Resolve country mentions, aggregate country visibility, build GDP comparison tables. | `explicit_geo_match.py`, `build_country_figure_tables.py`, `build_gdp_representation_v2.py` |
| [`documentation_audit/`](documentation_audit/) | Recompute two-annotator agreement from released score sheets. | `compute_documentation_audit_agreement.py` |
| [`sensitivity/`](sensitivity/) | Score paired answer variants and summarize controls. | `run_paired_official_scoring.py`, `run_truthfulqa_generation_similarity.py` |

The root [README](../README.md) has runnable commands for checks and paired scoring. Complete convention and geographic reruns also need upstream corpora and intermediate files; see [data sources](../DATA_SOURCES.md). Third-party DROP evaluator code and its license are under [`sensitivity/third_party/`](sensitivity/third_party/).
