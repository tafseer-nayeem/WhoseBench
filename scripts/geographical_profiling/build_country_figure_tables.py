from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ENGLISH_DIR = ROOT / "outputs" / "tables" / "english_core_suite"
TABLE_DIR = ROOT / "outputs" / "tables" / "country_figures"

FINAL_BENCHMARKS = [
    ("datacomp_1b_20k", "DataComp1B"),
    ("wit_english_50k", "WIT English"),
    ("redcaps_20k", "RedCaps"),
    ("cc12m_20k", "CC12M"),
    ("coyo_700m_20k", "COYO-700M"),
]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_top_by_benchmark(benchmark_rows: dict[str, list[dict[str, str]]], n: int) -> None:
    top_rows = []
    for _, benchmark_name in FINAL_BENCHMARKS:
        rows = benchmark_rows[benchmark_name][:n]
        for rank, row in enumerate(rows, start=1):
            top_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "rank": rank,
                    "country_alpha2": row["country_alpha2"],
                    "country_name": row["country_name"],
                    "share_resolved_pct": row["share_resolved_pct"],
                }
            )
    write_csv(
        TABLE_DIR / f"top{n}_by_benchmark.csv",
        top_rows,
        ["benchmark_name", "rank", "country_alpha2", "country_name", "share_resolved_pct"],
    )


def write_bottom_intersection(benchmark_maps: dict[str, dict[str, dict[str, str]]], n: int) -> None:
    intersections = None
    for _, benchmark_name in FINAL_BENCHMARKS:
        codes = set(benchmark_maps[benchmark_name].keys())
        intersections = codes if intersections is None else intersections & codes
    intersection_codes = sorted(intersections or [])

    bottom_candidates = []
    for alpha2 in intersection_codes:
        shares = [float(benchmark_maps[benchmark_name][alpha2]["share_resolved_pct"]) for _, benchmark_name in FINAL_BENCHMARKS]
        avg_share = sum(shares) / len(shares)
        country_name = benchmark_maps[FINAL_BENCHMARKS[0][1]][alpha2]["country_name"]
        bottom_candidates.append(
            {
                "country_alpha2": alpha2,
                "country_name": country_name,
                "average_share_pct": avg_share,
            }
        )

    selected_bottom = sorted(bottom_candidates, key=lambda row: row["average_share_pct"])[:n]
    selected_codes = [row["country_alpha2"] for row in selected_bottom]
    selected_lookup = {row["country_alpha2"]: row for row in selected_bottom}

    bottom_summary_rows = []
    for row in selected_bottom:
        bottom_summary_rows.append(
            {
                "country_alpha2": row["country_alpha2"],
                "country_name": row["country_name"],
                "average_share_pct": f"{row['average_share_pct']:.4f}",
            }
        )
    write_csv(
        TABLE_DIR / f"bottom{n}_intersection_countries.csv",
        bottom_summary_rows,
        ["country_alpha2", "country_name", "average_share_pct"],
    )

    bottom_grouped_rows = []
    for alpha2 in selected_codes:
        for _, benchmark_name in FINAL_BENCHMARKS:
            row = benchmark_maps[benchmark_name][alpha2]
            bottom_grouped_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "country_alpha2": alpha2,
                    "country_name": selected_lookup[alpha2]["country_name"],
                    "share_resolved_pct": row["share_resolved_pct"],
                    "average_share_pct": f"{selected_lookup[alpha2]['average_share_pct']:.4f}",
                }
            )
    write_csv(
        TABLE_DIR / f"bottom{n}_intersection_grouped.csv",
        bottom_grouped_rows,
        ["benchmark_name", "country_alpha2", "country_name", "share_resolved_pct", "average_share_pct"],
    )


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_rows: dict[str, list[dict[str, str]]] = {}
    benchmark_maps: dict[str, dict[str, dict[str, str]]] = {}

    for key, benchmark_name in FINAL_BENCHMARKS:
        rows = load_csv(ENGLISH_DIR / f"{key}_country_summary.csv")
        benchmark_rows[benchmark_name] = rows
        benchmark_maps[benchmark_name] = {row["country_alpha2"]: row for row in rows}

    write_top_by_benchmark(benchmark_rows, 5)
    write_top_by_benchmark(benchmark_rows, 10)
    write_bottom_intersection(benchmark_maps, 5)
    write_bottom_intersection(benchmark_maps, 10)

    note_rows = [
        {"metric": "benchmark_count", "value": str(len(FINAL_BENCHMARKS))},
        {"metric": "benchmark_names", "value": "; ".join(name for _, name in FINAL_BENCHMARKS)},
        {"metric": "bottom_selection_definition", "value": "bottom countries selected from the intersection of countries that appear in all five benchmarks"},
        {"metric": "bottom_selection_ranking", "value": "countries ranked by average share across the five benchmarks"},
        {
            "metric": "intersection_country_count",
            "value": str(
                len(
                    set.intersection(
                        *[set(benchmark_maps[name].keys()) for _, name in FINAL_BENCHMARKS]
                    )
                )
            ),
        },
    ]
    write_csv(TABLE_DIR / "selection_note.csv", note_rows, ["metric", "value"])

    print(f"Wrote {TABLE_DIR / 'top5_by_benchmark.csv'}")
    print(f"Wrote {TABLE_DIR / 'top10_by_benchmark.csv'}")
    print(f"Wrote {TABLE_DIR / 'bottom5_intersection_countries.csv'}")
    print(f"Wrote {TABLE_DIR / 'bottom5_intersection_grouped.csv'}")
    print(f"Wrote {TABLE_DIR / 'bottom10_intersection_countries.csv'}")
    print(f"Wrote {TABLE_DIR / 'bottom10_intersection_grouped.csv'}")
    print(f"Wrote {TABLE_DIR / 'selection_note.csv'}")


if __name__ == "__main__":
    main()
