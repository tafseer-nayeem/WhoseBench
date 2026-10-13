from __future__ import annotations

import argparse
import csv
import random
import re
import string
from collections import Counter
from pathlib import Path


ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
MULTISPACE = re.compile(r"\s+")


def normalize_exact_match(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLES.sub(" ", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text


def exact_match(candidate: str, references: tuple[str, ...]) -> bool:
    c = normalize_exact_match(candidate)
    return any(c == normalize_exact_match(ref) for ref in references)


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = normalize_exact_match(candidate).split()
    reference_tokens = normalize_exact_match(reference).split()
    if not candidate_tokens or not reference_tokens:
        return float(candidate_tokens == reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)
    common = sum((candidate_counts & reference_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def pp(x: float) -> str:
    return f"{100.0 * x:.1f} pp"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def control_variants(answer: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    stripped = answer.strip()
    if not stripped:
        return variants
    variants.append(("extra_whitespace", f"  {stripped}  "))
    if stripped.upper() != stripped:
        variants.append(("uppercase", stripped.upper()))
    if not stripped.endswith("."):
        variants.append(("trailing_period", f"{stripped}."))
    # Keep this intentionally small and normalizer-oriented. These are negative
    # controls, not new semantic convention variants.
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for kind, variant in variants:
        if variant != answer and variant not in seen:
            out.append((kind, variant))
            seen.add(variant)
    return out


def is_applicable_answer_text_scorer(record: dict[str, str]) -> bool:
    return record["accepted_by_benchmark_scorer"] != "not_applicable"


def score_benchmark_control(record: dict[str, str], variant: str, refs: tuple[str, ...]) -> bool | None:
    if not is_applicable_answer_text_scorer(record):
        return None
    if record["benchmark"] == "DROP":
        max_f1 = max((token_f1(variant, ref) for ref in refs), default=0.0)
        return exact_match(variant, refs) or max_f1 >= 0.999999
    return exact_match(variant, refs)


def bootstrap_ci(values_by_cluster: dict[str, list[float]], samples: int, rng: random.Random) -> tuple[float, float]:
    keys = sorted(values_by_cluster)
    if not keys:
        return 0.0, 0.0
    means: list[float] = []
    for _ in range(samples):
        vals: list[float] = []
        for key in (rng.choice(keys) for _ in keys):
            vals.extend(values_by_cluster[key])
        means.append(mean(vals))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def summarize(records: list[dict[str, str]], score_field: str, bootstrap_samples: int, rng: random.Random) -> dict[str, str]:
    scores = [float(r[score_field] == "True") for r in records if r[score_field] != "not_applicable"]
    by_cluster: dict[str, list[float]] = {}
    for r in records:
        if r[score_field] == "not_applicable":
            continue
        key = f"{r['benchmark']}::{r['split']}::{r['item_id']}"
        by_cluster.setdefault(key, []).append(float(r[score_field] == "True"))
    accept = mean(scores)
    lo, hi = bootstrap_ci(by_cluster, bootstrap_samples, rng)
    return {
        "n_controls": str(len(scores)),
        "accept_rate": f"{accept:.6f}",
        "rejection_rate": f"{1.0 - accept:.6f}",
        "accept_ci_low": f"{lo:.6f}",
        "accept_ci_high": f"{hi:.6f}",
    }


def write_markdown(summary_rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Scorer Negative-Control Sensitivity",
        "",
        "This experiment contrasts convention variants with simple surface controls that common benchmark normalizers are expected to accept: extra whitespace, uppercase, and trailing punctuation.",
        "",
        "| Scope | Group | Controls | Accepted [95% CI] | Rejected |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["scope"],
                    row["group"],
                    row["n_controls"],
                    f"{pct(float(row['accept_rate']))} [{pct(float(row['accept_ci_low']))}, {pct(float(row['accept_ci_high']))}]",
                    pct(float(row["rejection_rate"])),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation: these controls test whether the scorer fails any textual surface change. High acceptance for controls, alongside low scores for convention variants, supports the claim that the convention-variant result is not merely generic punctuation/case brittleness.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/exact_match_variant_records.csv"),
    )
    parser.add_argument(
        "--output-records",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/scorer_control_records.csv"),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/scorer_control_summary.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/SCORER_CONTROL_SENSITIVITY.md"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    with args.records.open(newline="", encoding="utf-8") as f:
        source_records = list(csv.DictReader(f))

    unique: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for record in source_records:
        key = (record["benchmark"], record["split"], record["item_id"], record["original_answer"])
        unique.setdefault(key, record)

    control_records: list[dict[str, str]] = []
    for record in unique.values():
        refs = tuple(record["references_joined"].split(" || "))
        for control_type, control_answer in control_variants(record["original_answer"]):
            benchmark_accept = score_benchmark_control(record, control_answer, refs)
            control_records.append(
                {
                    "benchmark": record["benchmark"],
                    "split": record["split"],
                    "item_id": record["item_id"],
                    "control_type": control_type,
                    "original_answer": record["original_answer"],
                    "control_answer": control_answer,
                    "accepted_by_standard_exact_match": str(exact_match(control_answer, refs)),
                    "accepted_by_applicable_answer_text_scorer": (
                        "not_applicable" if benchmark_accept is None else str(bool(benchmark_accept))
                    ),
                }
            )

    args.output_records.parent.mkdir(parents=True, exist_ok=True)
    with args.output_records.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(control_records[0]))
        writer.writeheader()
        writer.writerows(control_records)

    rng = random.Random(args.seed)
    summary_rows: list[dict[str, str]] = []
    for scope, field in (
        ("standard_exact_match", "accepted_by_standard_exact_match"),
        ("applicable_answer_text_scorer", "accepted_by_applicable_answer_text_scorer"),
    ):
        row = summarize(control_records, field, args.bootstrap_samples, rng)
        row.update({"scope": scope, "group": "ALL"})
        summary_rows.append(row)
        for control_type in sorted({r["control_type"] for r in control_records}):
            group_records = [r for r in control_records if r["control_type"] == control_type]
            row = summarize(group_records, field, args.bootstrap_samples, rng)
            row.update({"scope": scope, "group": control_type})
            summary_rows.append(row)

    with args.output_summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(summary_rows, args.output_md)
    print(f"wrote {args.output_records}")
    print(f"wrote {args.output_summary}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
