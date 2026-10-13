from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGLISH_DIR = ROOT / "outputs" / "tables" / "english_core_suite"
MULTI_DIR = ROOT / "outputs" / "tables" / "multilingual_suite"
FINAL_DIR = ROOT / "outputs" / "tables" / "final_country_figures"
MIN_COUNTRY_SAMPLES_PER_BENCHMARK = 2

ACTIVE_BENCHMARKS = [
    ("conceptual_captions_20k", "Conceptual Captions", ENGLISH_DIR),
    ("datacomp_1b_20k", "DataComp1B", ENGLISH_DIR),
    ("wit_english_50k", "WIT English", ENGLISH_DIR),
    ("redcaps_20k", "RedCaps", ENGLISH_DIR),
    ("cc12m_20k", "CC12M", ENGLISH_DIR),
    ("coyo_700m_20k", "COYO-700M", ENGLISH_DIR),
    ("wit_multilingual_merged", "WIT Multilingual", MULTI_DIR),
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    country_by_benchmark: dict[str, dict[str, float]] = {}
    rows_by_benchmark: dict[str, list[dict[str, str]]] = {}
    country_names: dict[str, str] = {}
    benchmark_presence: Counter[str] = Counter()
    total_sample_count: Counter[str] = Counter()

    for key, _, base_dir in ACTIVE_BENCHMARKS:
        rows = load_csv(base_dir / f"{key}_country_summary.csv")
        rows_by_benchmark[key] = rows
        benchmark_map: dict[str, float] = {}
        for row in rows:
            alpha2 = row["country_alpha2"]
            share = float(row["share_resolved_pct"])
            benchmark_map[alpha2] = share
            country_names[alpha2] = row["country_name"]
            benchmark_presence[alpha2] += 1
            total_sample_count[alpha2] += int(row["count"])
        country_by_benchmark[key] = benchmark_map

    average_share: dict[str, float] = {}
    total_share: dict[str, float] = {}
    nonzero_avg_share: dict[str, float] = {}

    for alpha2 in country_names:
        shares = [country_by_benchmark[key].get(alpha2, 0.0) for key, _, _ in ACTIVE_BENCHMARKS]
        nonzero = [share for share in shares if share > 0.0]
        average_share[alpha2] = sum(shares) / len(ACTIVE_BENCHMARKS)
        total_share[alpha2] = sum(shares)
        nonzero_avg_share[alpha2] = (sum(nonzero) / len(nonzero)) if nonzero else 0.0

    min_total_samples_for_bottom = 2

    top_rows = []
    for alpha2, avg in sorted(average_share.items(), key=lambda item: item[1], reverse=True)[:10]:
        top_rows.append(
            {
                "country_alpha2": alpha2,
                "country_name": country_names[alpha2],
                "average_share_pct": f"{avg:.4f}",
                "total_share_pct": f"{total_share[alpha2]:.4f}",
                "benchmark_presence": benchmark_presence[alpha2],
                "total_sample_count": total_sample_count[alpha2],
                "nonzero_average_share_pct": f"{nonzero_avg_share[alpha2]:.4f}",
            }
        )

    bottom_candidates = [
        alpha2
        for alpha2, count in total_sample_count.items()
        if count >= min_total_samples_for_bottom
    ]
    bottom_rows = []
    for alpha2 in sorted(bottom_candidates, key=lambda code: average_share[code])[:10]:
        bottom_rows.append(
            {
                "country_alpha2": alpha2,
                "country_name": country_names[alpha2],
                "average_share_pct": f"{average_share[alpha2]:.4f}",
                "total_share_pct": f"{total_share[alpha2]:.4f}",
                "benchmark_presence": benchmark_presence[alpha2],
                "total_sample_count": total_sample_count[alpha2],
                "nonzero_average_share_pct": f"{nonzero_avg_share[alpha2]:.4f}",
            }
        )

    for name, rows in [("top10_countries.csv", top_rows), ("bottom10_countries.csv", bottom_rows)]:
        with (FINAL_DIR / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "country_alpha2",
                    "country_name",
                    "average_share_pct",
                    "total_share_pct",
                    "benchmark_presence",
                    "total_sample_count",
                    "nonzero_average_share_pct",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

    summary_path = FINAL_DIR / "active_benchmark_country_extremes_summary.csv"
    summary_rows = []
    for alpha2 in sorted(country_names):
        summary_rows.append(
            {
                "country_alpha2": alpha2,
                "country_name": country_names[alpha2],
                "average_share_pct": f"{average_share[alpha2]:.4f}",
                "total_share_pct": f"{total_share[alpha2]:.4f}",
                "benchmark_presence": benchmark_presence[alpha2],
                "total_sample_count": total_sample_count[alpha2],
                "nonzero_average_share_pct": f"{nonzero_avg_share[alpha2]:.4f}",
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "country_alpha2",
                "country_name",
                "average_share_pct",
                "total_share_pct",
                "benchmark_presence",
                "total_sample_count",
                "nonzero_average_share_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    grouped_path = FINAL_DIR / "benchmarkwise_country_shares.csv"
    grouped_rows = []
    for selected_set, rows in (("top10", top_rows), ("bottom10", bottom_rows)):
        for row in rows:
            alpha2 = row["country_alpha2"]
            for key, benchmark_name, _ in ACTIVE_BENCHMARKS:
                grouped_rows.append(
                    {
                        "selected_set": selected_set,
                        "country_alpha2": alpha2,
                        "country_name": country_names[alpha2],
                        "benchmark_key": key,
                        "benchmark_name": benchmark_name,
                        "share_pct": f"{country_by_benchmark[key].get(alpha2, 0.0):.4f}",
                        "benchmark_presence": benchmark_presence[alpha2],
                        "total_sample_count": total_sample_count[alpha2],
                    }
                )
    with grouped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selected_set",
                "country_alpha2",
                "country_name",
                "benchmark_key",
                "benchmark_name",
                "share_pct",
                "benchmark_presence",
                "total_sample_count",
            ],
        )
        writer.writeheader()
        writer.writerows(grouped_rows)

    threshold_path = FINAL_DIR / "bottom_country_threshold_note.csv"
    with threshold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "active_benchmark_count", "value": len(ACTIVE_BENCHMARKS)})
        writer.writerow({"metric": "bottom_min_total_samples", "value": min_total_samples_for_bottom})
        writer.writerow(
            {
                "metric": "bottom_threshold_definition",
                "value": "country must have at least two resolved samples across active benchmark suites; no benchmark-overlap requirement is applied",
            }
        )

    benchmarkwise_bottom_rows = []
    benchmarkwise_bottom_wide_rows = []
    bottom_by_benchmark: dict[str, list[dict[str, str]]] = {}
    for key, benchmark_name, _ in ACTIVE_BENCHMARKS:
        eligible = [
            row
            for row in rows_by_benchmark[key]
            if int(row["count"]) >= MIN_COUNTRY_SAMPLES_PER_BENCHMARK
        ]
        selected = sorted(
            eligible,
            key=lambda row: (float(row["share_resolved_pct"]), row["country_name"]),
        )[:10]
        bottom_by_benchmark[key] = selected
        for rank, row in enumerate(selected, start=1):
            benchmarkwise_bottom_rows.append(
                {
                    "benchmark_key": key,
                    "benchmark_name": benchmark_name,
                    "rank": rank,
                    "country_alpha2": row["country_alpha2"],
                    "country_name": row["country_name"],
                    "count": row["count"],
                    "share_resolved_pct": row["share_resolved_pct"],
                }
            )

    for rank in range(1, 11):
        wide_row = {"rank": rank}
        for key, benchmark_name, _ in ACTIVE_BENCHMARKS:
            selected = bottom_by_benchmark[key]
            if rank <= len(selected):
                row = selected[rank - 1]
                wide_row[benchmark_name] = f"{row['country_name']} (n={row['count']})"
            else:
                wide_row[benchmark_name] = ""
        benchmarkwise_bottom_wide_rows.append(wide_row)

    benchmarkwise_bottom_path = FINAL_DIR / "benchmarkwise_bottom10_min2.csv"
    with benchmarkwise_bottom_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark_key",
                "benchmark_name",
                "rank",
                "country_alpha2",
                "country_name",
                "count",
                "share_resolved_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(benchmarkwise_bottom_rows)

    benchmarkwise_bottom_wide_path = FINAL_DIR / "benchmarkwise_bottom10_min2_wide.csv"
    with benchmarkwise_bottom_wide_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["rank"] + [benchmark_name for _, benchmark_name, _ in ACTIVE_BENCHMARKS],
        )
        writer.writeheader()
        writer.writerows(benchmarkwise_bottom_wide_rows)

    print(f"Wrote {FINAL_DIR / 'top10_countries.csv'}")
    print(f"Wrote {FINAL_DIR / 'bottom10_countries.csv'}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {grouped_path}")
    print(f"Wrote {threshold_path}")
    print(f"Wrote {benchmarkwise_bottom_path}")
    print(f"Wrote {benchmarkwise_bottom_wide_path}")


if __name__ == "__main__":
    main()
