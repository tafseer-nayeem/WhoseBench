from __future__ import annotations

import csv
import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CRITERIA = [
    ("source_user_context", "Source-user context"),
    ("target_user_context", "Target/eval. context"),
    ("annotator_visibility", "Annotator visibility"),
    ("locale_assumptions", "Locale assumptions"),
    ("evaluation_normalization", "Evaluation normalization"),
    ("geographic_metadata", "Geographic metadata"),
]

DISAGREEMENT_NOTES = {
    ("CC12M", "geographic_metadata"): (
        "CC12M exposes image URLs in the official TSV format; because URL metadata "
        "is part of the rubric, the adjudicated score is 2."
    )
}


def read_scores(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def cohen_kappa(a: list[int], b: list[int]) -> tuple[float, float | None]:
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    counts_a = Counter(a)
    counts_b = Counter(b)
    expected = sum((counts_a[c] / n) * (counts_b[c] / n) for c in (0, 1, 2))
    if abs(1 - expected) < 1e-12:
        return observed, None
    return observed, (observed - expected) / (1 - expected)


def build_latex_table(summary_rows: list[dict[str, object]]) -> str:
    lines = [
        "% Documentation audit agreement table.",
        "% Recommended packages: booktabs.",
        "",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Two-annotator agreement for the documentation audit before adjudication. Agreement is computed over 16 resources for each criterion. Cohen's $\kappa$ is omitted when both annotators assign a constant score, making chance-corrected agreement undefined.}",
        r"\label{tab:documentation_audit_agreement}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"\textbf{Criterion} & \textbf{Exact agree.} & \textbf{Cohen's $\kappa$} & \textbf{Disagr.} \\",
        r" & \textbf{(\%)} & & \\",
        r"\midrule",
    ]
    for row in summary_rows:
        if row["criterion_id"] == "overall_micro":
            lines.append(r"\midrule")
        lines.append(
            f"{row['criterion']} & {row['exact_agreement_pct']} & "
            f"{row['cohen_kappa']} & {row['disagreements']} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute two-annotator documentation audit agreement.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "documentation_audit",
        help="Directory containing documentation_audit_annotator1_scores.csv and documentation_audit_annotator2_scores.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "documentation_audit",
        help="Directory where agreement CSVs should be written.",
    )
    parser.add_argument(
        "--latex-dir",
        type=Path,
        default=None,
        help="Optional directory where the LaTeX agreement table should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table_dir = args.input_dir
    output_dir = args.output_dir
    writing_dir = args.latex_dir

    annotator1 = read_scores(table_dir / "documentation_audit_annotator1_scores.csv")
    annotator2 = read_scores(table_dir / "documentation_audit_annotator2_scores.csv")
    if [row["resource"] for row in annotator1] != [row["resource"] for row in annotator2]:
        raise ValueError("Annotator score sheets must contain resources in the same order.")

    summary_rows: list[dict[str, object]] = []
    all_a: list[int] = []
    all_b: list[int] = []

    for criterion_id, criterion_name in CRITERIA:
        a = [int(row[criterion_id]) for row in annotator1]
        b = [int(row[criterion_id]) for row in annotator2]
        observed, kappa = cohen_kappa(a, b)
        all_a.extend(a)
        all_b.extend(b)
        summary_rows.append(
            {
                "criterion_id": criterion_id,
                "criterion": criterion_name,
                "n_items": len(a),
                "exact_agreement_pct": f"{100 * observed:.1f}",
                "cohen_kappa": "--" if kappa is None else f"{kappa:.3f}",
                "disagreements": sum(x != y for x, y in zip(a, b)),
            }
        )

    observed, kappa = cohen_kappa(all_a, all_b)
    summary_rows.append(
        {
            "criterion_id": "overall_micro",
            "criterion": "Overall (micro)",
            "n_items": len(all_a),
            "exact_agreement_pct": f"{100 * observed:.1f}",
            "cohen_kappa": "--" if kappa is None else f"{kappa:.3f}",
            "disagreements": sum(x != y for x, y in zip(all_a, all_b)),
        }
    )

    write_csv(
        output_dir / "documentation_audit_agreement_summary.csv",
        summary_rows,
        ["criterion_id", "criterion", "n_items", "exact_agreement_pct", "cohen_kappa", "disagreements"],
    )

    disagreement_rows: list[dict[str, object]] = []
    for row_a, row_b in zip(annotator1, annotator2):
        for criterion_id, criterion_name in CRITERIA:
            if row_a[criterion_id] != row_b[criterion_id]:
                disagreement_rows.append(
                    {
                        "resource": row_a["resource"],
                        "criterion": criterion_name,
                        "annotator1_score": row_a[criterion_id],
                        "annotator2_score": row_b[criterion_id],
                        "adjudication_note": DISAGREEMENT_NOTES.get((row_a["resource"], criterion_id), ""),
                    }
                )
    write_csv(
        output_dir / "documentation_audit_disagreements.csv",
        disagreement_rows,
        ["resource", "criterion", "annotator1_score", "annotator2_score", "adjudication_note"],
    )

    if writing_dir is not None:
        writing_dir.mkdir(parents=True, exist_ok=True)
        (writing_dir / "documentation_audit_agreement_table.tex").write_text(
            build_latex_table(summary_rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
