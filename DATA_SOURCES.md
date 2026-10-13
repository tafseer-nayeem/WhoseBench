# Data sources and use boundaries

The release contains detector rules, country aliases, documentation records,
item-level sensitivity records, aggregate tables, and figures. It does not
redistribute complete benchmark questions, image files, caption corpora, or
private data. The original benchmark sources and download records are listed
under data/benchmark_sources/. The acquisition scripts are under
scripts/convention/ and scripts/geographical_profiling/.

## Convention audit

The text-benchmark tables summarize matched spelling,
date, time, unit, and related convention signals. The source tables identify
upstream repositories and available revisions. Full extraction needs the
original benchmark data and may change if upstream revisions are not the same.

## Geographic audit

The included country lexicon and resolver code operate on
caption or metadata text from public image-text resources. Included tables
contain derived country counts, shares, World Bank joins, correlations, and
regression summaries. Raw captions and images are not bundled. A new lexicon
built from current country-name packages may differ from the included one;
use the included CSV when checking released results.

## Documentation audit

The rubric, evidence anchors, annotation sheets, and
adjudicated scores are included under data/documentation_audit/. Source URLs
identify the papers, dataset cards, and repositories reviewed. The template
there is a reusable reporting aid, not a dataset of observations.

## Scoring sensitivity

`results/evaluation_sensitivity_broad/` contains paired
answer-variant records and separate DROP exact-match/F1 and TruthfulQA
reference-similarity summaries. DROP uses the bundled AllenNLP evaluator;
the source attribution and license are under scripts/sensitivity/third_party/.
These are scorer-side counterfactuals, not measured model-accuracy changes.

The root LICENSE covers release code. Upstream corpora, source papers, and
third-party code retain their own licenses and terms. A local validation run
checks selected released values. It is not a substitute for obtaining and
reprocessing every upstream corpus.
