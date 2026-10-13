# Whose Data, Whose Benchmark?

Anonymous research release

This is the anonymous release of the user-centric evaluation-resource audit.
It includes the audit code, reusable resources, annotation records, derived
results, and figures for the three analyses: convention signals in text
benchmarks, geographic visibility in image-text resources, and documentation
of user assumptions across both groups. It also includes the paired scoring
sensitivity extension. The original benchmark corpora are not redistributed.

## Contents

- `data/audit_taxonomy.csv`: the seven audit dimensions and their scope.
- `data/convention_detector_inventory/`: convention inventories and tests.
- `data/country_resolver_lexicons/`: the country resolver's input lexicon.
- `data/documentation_audit/`: rubric, evidence anchors, two annotator score sheets, adjudicated scores, agreement, and a reusable template.
- `data/benchmark_sources/`: resource selection, source references, and download coverage records.
- `scripts/`: acquisition, detection, country resolution, aggregation, statistical analysis, documentation agreement, and scoring checks.
- `results/convention/`: text-audit tables and PDF figures.
- `results/geographical_profiling/`: geographic and GDP-analysis tables and PDF figures.
- `results/evaluation_sensitivity_broad/`: item-level paired scoring and summary tables, with DROP and TruthfulQA metrics kept separate.
- `DATA_SOURCES.md`: provenance, licensing, and rerun boundaries.

The six documentation fields in the scoring rubric are source-user context,
target-user context, annotator visibility, locale assumptions, evaluation
normalization, and geographic metadata. The nine-field template in
data/documentation_audit/documentation_template.csv is a reporting aid for
new evaluation resources; it is not a new score or an additional experiment.

## Install

Python 3.10 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Check the included results:

```bash
.venv/bin/python scripts/validate_outputs.py
```

Recompute the documentation-audit agreement:

```bash
.venv/bin/python scripts/documentation_audit/compute_documentation_audit_agreement.py \
    --input-dir data/documentation_audit --output-dir /tmp/documentation_audit_check
```

Recompute the answer-variant sensitivity analyses:

```bash
.venv/bin/python scripts/sensitivity/run_paired_official_scoring.py \
    --output-dir /tmp/paired_scoring
.venv/bin/python scripts/sensitivity/run_truthfulqa_generation_similarity.py \
    --records /tmp/paired_scoring/truthfulqa_reference_diagnostic_records.csv \
    --output-dir /tmp/truthfulqa_similarity
```

The paired scoring script retrieves the original DROP annotations and uses
the bundled AllenNLP evaluator, whose license and source attribution are in
scripts/sensitivity/third_party/. TruthfulQA reference data are loaded at a
pinned revision. DROP exact match and F1 are separate metrics; the TruthfulQA
generation-similarity comparison is a reference-side counterfactual, not a
measured change in model accuracy. The item-level CSV files are under
results/evaluation_sensitivity_broad/.

Use the included lexicon to reproduce the released results. Rebuilding it
with current Babel or GeoNames data can change entries; its optional builder
requires --output-csv so it does not overwrite the included lexicon.
Full convention and geographic reruns require acquiring the source corpora
and recreating intermediate inputs. scripts/validate_outputs.py checks
selected reported values, not every row or every upstream extraction step.

The manuscript PDF is not included in this archive. Source URLs are retained
for provenance. Standalone figure scripts and intermediate reports are not
part of this package.
