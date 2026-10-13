# Documentation audit

Two annotators reviewed public papers, dataset cards, and repositories for all **16 resources**. Each of six criteria is scored **0** (absent), **1** (explicitly mentioned), or **2** (operationalized through inspectable evidence). The [rubric](documentation_audit_rubric.csv), [evidence anchors](documentation_audit_evidence_anchors.csv), [two score sheets](documentation_audit_annotator1_scores.csv), [adjudicated scores](documentation_audit_adjudicated_scores.csv), and [agreement summary](documentation_audit_agreement_summary.csv) are included.

| Criterion | Mean / 2 | Mentioned (1 or 2) | Operationalized (2) |
| --- | ---: | ---: | ---: |
| Source-user context | 2.00 | 100.0% | 100.0% |
| Target-user / evaluation context | 1.00 | 100.0% | 0.0% |
| Annotator visibility | 0.56 | 56.2% | 0.0% |
| Locale assumptions | 0.31 | 25.0% | 6.2% |
| Evaluation normalization | 1.44 | 75.0% | 68.8% |
| Source / geographic metadata | 0.69 | 37.5% | 31.2% |

These values are from the [released summary CSV](documentation_audit_summary.csv). The [project site](https://tafseer-nayeem.github.io/WhoseBench/#documentation) gives the audit questions and examples. In particular, every resource identifies a content source, but none operationalizes the target-user or evaluation context.

To recompute inter-annotator agreement without modifying released files:

```bash
python3 scripts/documentation_audit/compute_documentation_audit_agreement.py \
  --input-dir data/documentation_audit --output-dir /tmp/whosebench_agreement
```

Run this from the repository root. The [nine-field documentation template](documentation_template.csv) is a reporting aid for new resources; it is not a scored seventh criterion or an additional experiment.
