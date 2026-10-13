# Whose Data, Whose Benchmark?

**A user-centric framework for auditing evaluation resources**

[Project website](https://tafseer-nayeem.github.io/WhoseBench/) | [Code repository](https://github.com/tafseer-nayeem/WhoseBench)

WhoseBench examines whose conventions, places, and intended users are represented in evaluation resources. The study audits **11 text benchmarks** and **five image-text resources** across seven dimensions. This repository contains the audit code, detector and country lexicons, documentation annotations, derived tables, and figures. It does **not** redistribute the original benchmark corpora or images.

## Findings at a glance

| Analysis | Released finding | Explore |
| --- | --- | --- |
| Text conventions | American-centered forms make up 62.4% to 94.4% of *matched signals* across the 11 text benchmarks. | [Convention results](results/convention/README.md) |
| Scoring sensitivity | On 12 eligible DROP answer pairs, exact match falls from 100.0% to 33.3% after meaning-preserving variants; token F1 falls to 77.2%. | [Paired scoring results](results/evaluation_sensitivity_broad/README.md) |
| Geographic visibility | The US leads resolved textual country references in four of five image-text resources; Iceland leads CC12M. | [Geographic results](results/geographical_profiling/README.md) |
| Documentation | All 16 resources operationalize source-user context; none operationalize target-user context. | [Documentation audit](data/documentation_audit/README.md) |

## Repository map

| Path | Contents |
| --- | --- |
| [`data/audit_taxonomy.csv`](data/audit_taxonomy.csv) | Seven dimensions, questions, and operational signals. |
| [`data/benchmark_sources/`](data/benchmark_sources/) | Resource selection, source references, and acquisition coverage. |
| [`data/convention_detector_inventory/`](data/convention_detector_inventory/) | Convention inventories and tests. |
| [`data/country_resolver_lexicons/`](data/country_resolver_lexicons/) | Released country lexicon used by the geographic audit. |
| [`data/documentation_audit/`](data/documentation_audit/README.md) | Rubric, evidence anchors, two annotator sheets, adjudication, agreement, and reusable template. |
| [`results/convention/`](results/convention/README.md) | Text-audit tables and PDF figures. |
| [`results/geographical_profiling/`](results/geographical_profiling/README.md) | Country visibility and GDP-association tables and figures. |
| [`results/evaluation_sensitivity_broad/`](results/evaluation_sensitivity_broad/README.md) | Item-level answer pairs and benchmark-specific scorer summaries. |
| [`scripts/`](scripts/README.md) | Acquisition, detection, aggregation, agreement, scoring, and validation code. |
| [`DATA_SOURCES.md`](DATA_SOURCES.md) | Provenance, licensing, and rerun boundaries. |

## Quick start

Python 3.10 or newer is required. The included-output validator uses only the standard library, so it can run before installing optional analysis dependencies:

```bash
python3 scripts/validate_outputs.py
```

For the analysis scripts:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Recompute documentation agreement from the released annotation sheets:

```bash
.venv/bin/python scripts/documentation_audit/compute_documentation_audit_agreement.py \
  --input-dir data/documentation_audit --output-dir /tmp/whosebench_agreement
```

Recompute the paired scoring analysis (downloads pinned upstream reference data):

```bash
.venv/bin/python scripts/sensitivity/run_paired_official_scoring.py \
  --output-dir /tmp/whosebench_paired_scoring
.venv/bin/python scripts/sensitivity/run_truthfulqa_generation_similarity.py \
  --records /tmp/whosebench_paired_scoring/truthfulqa_reference_diagnostic_records.csv \
  --output-dir /tmp/whosebench_truthfulqa_similarity
```

The DROP script uses the bundled AllenNLP evaluator, with attribution and license in [`scripts/sensitivity/third_party/`](scripts/sensitivity/third_party/). DROP exact match and token F1 are different measures; the TruthfulQA similarity experiment is a **reference-side counterfactual**, not a measured change in model accuracy. Full convention and geographic reruns require acquiring the upstream corpora and rebuilding intermediate inputs. The local validator checks selected released values, not every upstream extraction step.

Use the included country lexicon to reproduce released outputs; rebuilding it with newer Babel or GeoNames data can change entries. The optional builder requires `--output-csv` and will not overwrite the included lexicon by default. See [`DATA_SOURCES.md`](DATA_SOURCES.md) for source and licensing details.
