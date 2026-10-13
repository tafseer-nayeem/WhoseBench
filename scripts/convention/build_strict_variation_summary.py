from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "analysis"


KEEP_INVENTORY_CATEGORIES = {"Orthographic/Spelling"}
KEEP_SOURCES = {"convention"}


def keep_hit(hit: dict) -> bool:
    if hit.get("source") in KEEP_SOURCES:
        return True
    return hit.get("category") in KEEP_INVENTORY_CATEGORIES


def classify(american_count: int, other_count: int) -> str:
    if american_count and other_count:
        return "mixed"
    if american_count:
        return "american_only"
    if other_count:
        return "other_only"
    return "no_match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a strict benchmark variation summary from a scan output.")
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix matching a tagged scan output, e.g. provider_expanded.",
    )
    return parser.parse_args()


def tagged_output_path(name: str, tag: str) -> Path:
    path = OUTPUT_DIR / name
    if not tag:
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def main() -> None:
    args = parse_args()
    filtered_path = tagged_output_path("filtered_variation_samples.csv", args.tag)
    summary_path = tagged_output_path("variation_summary_by_benchmark.csv", args.tag)

    total_items = {}
    with summary_path.open() as handle:
        for row in csv.DictReader(handle):
            total_items[(row["benchmark_family"], row["benchmark_task"])] = int(row["total_items"])

    benchmark_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    overall = Counter()
    term_counter = Counter()

    with filtered_path.open() as handle:
        for row in csv.DictReader(handle):
            family = row["benchmark_family"]
            task = row["benchmark_task"]
            american_hits = [hit for hit in json.loads(row["american_matches"]) if keep_hit(hit)]
            other_hits = [hit for hit in json.loads(row["other_matches"]) if keep_hit(hit)]
            if not american_hits and not other_hits:
                continue

            american_count = len(american_hits)
            other_count = len(other_hits)
            classification = classify(american_count, other_count)

            overall["flagged_items"] += 1
            overall["american_match_count"] += american_count
            overall["other_match_count"] += other_count
            overall[classification] += 1

            stats = benchmark_stats[(family, task)]
            stats["flagged_items"] += 1
            stats["american_match_count"] += american_count
            stats["other_match_count"] += other_count
            stats[classification] += 1

            for hit in american_hits:
                term_counter[(family, task, "american", hit["category"], hit["term"].lower())] += 1
            for hit in other_hits:
                term_counter[(family, task, "other", hit["category"], hit["term"].lower())] += 1

    summary_rows = []
    for key, total in sorted(total_items.items()):
        family, task = key
        stats = benchmark_stats[key]
        flagged = stats["flagged_items"]
        american_total = stats["american_match_count"]
        other_total = stats["other_match_count"]
        total_matches = american_total + other_total
        summary_rows.append(
            {
                "benchmark_family": family,
                "benchmark_task": task,
                "total_items": str(total),
                "flagged_items": str(flagged),
                "flagged_pct": f"{(100 * flagged / total):.2f}" if total else "0.00",
                "american_only_items": str(stats["american_only"]),
                "other_only_items": str(stats["other_only"]),
                "mixed_items": str(stats["mixed"]),
                "american_match_count": str(american_total),
                "other_match_count": str(other_total),
                "american_match_pct": f"{(100 * american_total / total_matches):.2f}" if total_matches else "0.00",
                "other_match_pct": f"{(100 * other_total / total_matches):.2f}" if total_matches else "0.00",
            }
        )

    term_rows = []
    for (family, task, side, category, term), count in term_counter.most_common():
        term_rows.append(
            {
                "benchmark_family": family,
                "benchmark_task": task,
                "side": side,
                "category": category,
                "term": term,
                "count": str(count),
            }
        )

    strict_csv = tagged_output_path("variation_summary_strict_by_benchmark.csv", args.tag)
    with strict_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    strict_terms = tagged_output_path("variation_term_frequencies_strict.csv", args.tag)
    with strict_terms.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(term_rows[0]) if term_rows else [
            "benchmark_family", "benchmark_task", "side", "category", "term", "count"
        ])
        writer.writeheader()
        writer.writerows(term_rows)

    total_scanned = sum(total_items.values())
    total_matches = overall["american_match_count"] + overall["other_match_count"]
    lines = [
        "# Strict Benchmark Variation Scan Summary",
        "",
        "This stricter summary keeps only:",
        "",
        "- spelling-based inventory matches (`Orthographic/Spelling`)",
        "- explicit convention-pattern matches for date, time, and units",
        "",
        "It excludes the broader `Vocabulary` category because that category contains semantically ambiguous items that can overstate American-vs-other rates in technical or domain-specific contexts.",
        "",
        f"- Total benchmark items scanned: {total_scanned}",
        f"- Strict-flagged items: {overall['flagged_items']} ({(100 * overall['flagged_items'] / total_scanned):.2f}%)" if total_scanned else "- Strict-flagged items: 0",
        f"- American-only strict-flagged items: {overall['american_only']}",
        f"- Other-only strict-flagged items: {overall['other_only']}",
        f"- Mixed strict-flagged items: {overall['mixed']}",
        f"- American-side strict match share: {(100 * overall['american_match_count'] / total_matches):.2f}%" if total_matches else "- American-side strict match share: 0.00%",
        f"- Other-side strict match share: {(100 * overall['other_match_count'] / total_matches):.2f}%" if total_matches else "- Other-side strict match share: 0.00%",
        "",
        "## By benchmark",
        "",
        "| Benchmark | Task | Items | Strict-flagged | American only | Other only | Mixed | American match % | Other match % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark_family']} | {row['benchmark_task']} | {row['total_items']} | {row['flagged_items']} ({row['flagged_pct']}%) | {row['american_only_items']} | {row['other_only_items']} | {row['mixed_items']} | {row['american_match_pct']} | {row['other_match_pct']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Use this strict summary for benchmark-level comparisons with conservative matching.",
            "- Use the original summary when you want a broader exploratory view of lexical plus formatting variation.",
        ]
    )
    tagged_output_path("variation_summary_strict.md", args.tag).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
