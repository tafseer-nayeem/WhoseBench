from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def pp(x: float) -> str:
    return f"{100.0 * x:.1f} pp"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def original_accept_for_record(record: dict[str, str], scope: str) -> bool:
    """Original references are the scorer-aligned answer baseline.

    This script measures the paired score change caused by replacing an
    accepted reference answer with a conservative convention variant. The
    original answer is scored against the same reference set as a false-rejection
    baseline.
    """
    if scope == "all_standard_exact_match":
        return True
    if scope == "applicable_answer_text_scorer":
        return True
    raise ValueError(scope)


def variant_accept_for_record(record: dict[str, str], scope: str) -> bool | None:
    if scope == "all_standard_exact_match":
        return is_true(record["accepted_by_exact_match"])
    if scope == "applicable_answer_text_scorer":
        value = record["accepted_by_benchmark_scorer"]
        if value == "not_applicable":
            return None
        return is_true(value)
    raise ValueError(scope)


def score_delta(records: list[dict[str, str]], scope: str) -> tuple[float, float, float, float, int]:
    original_scores: list[float] = []
    variant_scores: list[float] = []
    for record in records:
        variant_accept = variant_accept_for_record(record, scope)
        if variant_accept is None:
            continue
        original_scores.append(float(original_accept_for_record(record, scope)))
        variant_scores.append(float(variant_accept))

    n = len(variant_scores)
    original_rate = mean(original_scores)
    variant_rate = mean(variant_scores)
    delta = variant_rate - original_rate
    rejection = 1.0 - variant_rate
    return original_rate, variant_rate, delta, rejection, n


def bootstrap_delta_ci(
    records: list[dict[str, str]],
    scope: str,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    valid_records = [r for r in records if variant_accept_for_record(r, scope) is not None]
    clusters: dict[str, list[dict[str, str]]] = {}
    for record in valid_records:
        key = f"{record['benchmark']}::{record['split']}::{record['item_id']}"
        clusters.setdefault(key, []).append(record)
    keys = sorted(clusters)
    if not keys:
        return 0.0, 0.0
    deltas: list[float] = []
    for _ in range(samples):
        sampled: list[dict[str, str]] = []
        for key in (rng.choice(keys) for _ in keys):
            sampled.extend(clusters[key])
        _, _, delta, _, _ = score_delta(sampled, scope)
        deltas.append(delta)
    deltas.sort()
    lo_idx = int(0.025 * (len(deltas) - 1))
    hi_idx = int(0.975 * (len(deltas) - 1))
    return deltas[lo_idx], deltas[hi_idx]


def summarize(
    records: list[dict[str, str]],
    scope: str,
    grouping: str,
    group: str,
    bootstrap_samples: int,
    rng: random.Random,
) -> dict[str, str]:
    original_rate, variant_rate, delta, rejection, n = score_delta(records, scope)
    ci_low, ci_high = bootstrap_delta_ci(records, scope, bootstrap_samples, rng)
    return {
        "scope": scope,
        "grouping": grouping,
        "group": group,
        "n_variants": str(n),
        "original_accept_rate": f"{original_rate:.6f}",
        "variant_accept_rate": f"{variant_rate:.6f}",
        "score_delta_variant_minus_original": f"{delta:.6f}",
        "score_delta_ci_low": f"{ci_low:.6f}",
        "score_delta_ci_high": f"{ci_high:.6f}",
        "rejection_rate": f"{rejection:.6f}",
    }


def grouped(records: list[dict[str, str]], key: str) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for record in records:
        out.setdefault(record[key], []).append(record)
    return out


def grouped_by_keys(records: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    out: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for record in records:
        group_key = tuple(record[key] for key in keys)
        out.setdefault(group_key, []).append(record)
    return out


def write_markdown(rows: list[dict[str, str]], output_path: Path) -> None:
    def row(scope: str, grouping: str, group: str) -> dict[str, str]:
        for item in rows:
            if item["scope"] == scope and item["grouping"] == grouping and item["group"] == group:
                return item
        raise KeyError((scope, grouping, group))

    def md_table(selected: list[dict[str, str]], include_group: bool = True) -> list[str]:
        if include_group:
            lines = [
                "| Group | Variants | Original-answer score | Valid-variant score | Score change [95% CI] | Valid-variant rejection |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        else:
            lines = [
                "| Scope | Variants | Original-answer score | Valid-variant score | Score change [95% CI] | Valid-variant rejection |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        for item in selected:
            label = item["group"] if include_group else item["scope"].replace("_", " ")
            lines.append(
                "| "
                + " | ".join(
                    [
                        label,
                        item["n_variants"],
                        pct(float(item["original_accept_rate"])),
                        pct(float(item["variant_accept_rate"])),
                        (
                            f"{pp(float(item['score_delta_variant_minus_original']))} "
                            f"[{pp(float(item['score_delta_ci_low']))}, "
                            f"{pp(float(item['score_delta_ci_high']))}]"
                        ),
                        pct(float(item["rejection_rate"])),
                    ]
                )
                + " |"
            )
        return lines

    headline = [
        row("all_standard_exact_match", "ALL", "ALL"),
        row("applicable_answer_text_scorer", "ALL", "ALL"),
    ]
    applicable_by_benchmark = [
        r
        for r in rows
        if r["scope"] == "applicable_answer_text_scorer"
        and r["grouping"] == "benchmark"
        and int(r["n_variants"]) > 0
    ]
    applicable_by_type = [
        r
        for r in rows
        if r["scope"] == "applicable_answer_text_scorer"
        and r["grouping"] == "perturbation_type"
        and int(r["n_variants"]) > 0
    ]
    broad_by_type = [
        r
        for r in rows
        if r["scope"] == "all_standard_exact_match"
        and r["grouping"] == "perturbation_type"
        and int(r["n_variants"]) > 0
    ]
    applicable_by_benchmark_type = [
        r
        for r in rows
        if r["scope"] == "applicable_answer_text_scorer"
        and r["grouping"] == "benchmark_x_perturbation_type"
        and int(r["n_variants"]) > 0
    ]
    broad_by_benchmark_type = [
        r
        for r in rows
        if r["scope"] == "all_standard_exact_match"
        and r["grouping"] == "benchmark_x_perturbation_type"
        and int(r["n_variants"]) > 0
    ]

    lines: list[str] = []
    lines.append("# Paired Evaluation-Sensitivity Score Deltas")
    lines.append("")
    lines.append(
        "A rejection rate alone does not measure the size of a score change. "
        "For each valid convention variant, we score the original reference answer and the "
        "valid variant against the same reference/scorer. The result is a paired score change."
    )
    lines.append("")
    lines.append("The original-answer score is the false-rejection baseline for unperturbed correct answers.")
    lines.append("")
    lines.extend(md_table(headline, include_group=False))
    lines.append("")
    lines.append("## Applicable Answer-Text Scorers by Benchmark")
    lines.append("")
    lines.extend(md_table(applicable_by_benchmark))
    lines.append("")
    lines.append("## Applicable Answer-Text Scorers by Perturbation Type")
    lines.append("")
    lines.extend(md_table(applicable_by_type))
    lines.append("")
    lines.append("## Applicable Answer-Text Scorers by Benchmark and Perturbation Type")
    lines.append("")
    lines.extend(md_table(applicable_by_benchmark_type))
    lines.append("")
    lines.append("## Broad Standard Exact-Match Diagnostic by Perturbation Type")
    lines.append("")
    lines.extend(md_table(broad_by_type))
    lines.append("")
    lines.append("## Broad Standard Exact-Match Diagnostic by Benchmark and Perturbation Type")
    lines.append("")
    lines.extend(md_table(broad_by_benchmark_type))
    lines.append("")
    lines.append(
        "Interpretation: this is a scorer-side evaluation outcome: the same correct answer, "
        "expressed as a conservative valid convention variant, receives a lower score. "
        "It is not a model-generation accuracy experiment."
    )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/exact_match_variant_records.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/score_delta_decomposition.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/SCORE_DELTA_DECOMPOSITION.md"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    with args.records.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    rows: list[dict[str, str]] = []
    rng = random.Random(args.seed)
    for scope in ("all_standard_exact_match", "applicable_answer_text_scorer"):
        rows.append(summarize(records, scope, "ALL", "ALL", args.bootstrap_samples, rng))
        for benchmark, group_records in sorted(grouped(records, "benchmark").items()):
            rows.append(summarize(group_records, scope, "benchmark", benchmark, args.bootstrap_samples, rng))
        for kind, group_records in sorted(grouped(records, "convention_type").items()):
            rows.append(summarize(group_records, scope, "perturbation_type", kind, args.bootstrap_samples, rng))
        for group_key, group_records in sorted(grouped_by_keys(records, ("benchmark", "convention_type")).items()):
            benchmark, kind = group_key
            rows.append(
                summarize(
                    group_records,
                    scope,
                    "benchmark_x_perturbation_type",
                    f"{benchmark} / {kind}",
                    args.bootstrap_samples,
                    rng,
                )
            )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(rows, args.output_md)
    print(f"wrote {args.output_csv}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
