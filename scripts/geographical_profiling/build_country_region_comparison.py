from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STRICT_DIR = ROOT / "outputs" / "tables"
EXPANDED_DIR = ROOT / "outputs" / "tables" / "country_region_expanded"
OUTPUT_DIR = ROOT / "outputs" / "tables" / "country_region_expanded"

DATASET_ORDER = [
    "wit_en_10k",
    "wit_fr_10k",
    "wit_de_10k",
    "wit_es_10k",
    "wit_ja_10k",
    "conceptual_captions_20k",
    "datacomp_1b_20k",
]

DISPLAY_NAMES = {
    "wit_en_10k": "WIT-en",
    "wit_fr_10k": "WIT-fr",
    "wit_de_10k": "WIT-de",
    "wit_es_10k": "WIT-es",
    "wit_ja_10k": "WIT-ja",
    "conceptual_captions_20k": "Conceptual Captions",
    "datacomp_1b_20k": "DataComp",
}


def read_metric_csv(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            data[row["metric"]] = row["value"]
    return data


def read_top_country(path: Path) -> tuple[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, None)
        if not first:
            return "", "0.0000"
        return first["country_name"], first["share_resolved_pct"]


def read_top_region(path: Path) -> tuple[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        first = next(reader, None)
        if not first:
            return "", "0.0000"
        return first["region"], first["share_resolved_pct"]


def build_cross_dataset_summary() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "cross_dataset_country_region_summary.csv"
    strict_compare_path = OUTPUT_DIR / "strict_vs_expanded_coverage_summary.csv"
    region_long_path = OUTPUT_DIR / "cross_dataset_region_long.csv"

    summary_rows: list[dict[str, str]] = []
    strict_rows: list[dict[str, str]] = []
    region_rows: list[dict[str, str]] = []

    for dataset in DATASET_ORDER:
        expanded_metrics = read_metric_csv(EXPANDED_DIR / f"{dataset}_metrics_summary.csv")
        strict_metrics = read_metric_csv(STRICT_DIR / f"{dataset}_match_summary.csv")
        top_country, top_country_share = read_top_country(EXPANDED_DIR / f"{dataset}_country_summary.csv")
        top_region, top_region_share = read_top_region(EXPANDED_DIR / f"{dataset}_region_summary.csv")

        total_items = int(expanded_metrics["total_items"])
        strict_resolved = (
            int(strict_metrics.get("status::single_country", "0"))
            + int(strict_metrics.get("status::majority_country", "0"))
        )
        expanded_resolved = int(expanded_metrics["resolved_items"])

        summary_rows.append(
            {
                "dataset": dataset,
                "display_name": DISPLAY_NAMES[dataset],
                "total_items": str(total_items),
                "resolved_items": expanded_metrics["resolved_items"],
                "resolved_coverage_pct": expanded_metrics["resolved_coverage_pct"],
                "western_share_resolved_pct": expanded_metrics["western_share_resolved_pct"],
                "top_country": top_country,
                "top_country_share_resolved_pct": top_country_share,
                "top3_country_share_resolved_pct": expanded_metrics["top3_country_share_resolved_pct"],
                "top_region": top_region,
                "top_region_share_resolved_pct": top_region_share,
            }
        )

        strict_rows.append(
            {
                "dataset": dataset,
                "display_name": DISPLAY_NAMES[dataset],
                "total_items": str(total_items),
                "strict_resolved_items": str(strict_resolved),
                "strict_resolved_coverage_pct": f"{(100.0 * strict_resolved / total_items) if total_items else 0.0:.4f}",
                "expanded_resolved_items": str(expanded_resolved),
                "expanded_resolved_coverage_pct": expanded_metrics["resolved_coverage_pct"],
                "coverage_gain_points": f"{((100.0 * expanded_resolved / total_items) - (100.0 * strict_resolved / total_items)) if total_items else 0.0:.4f}",
            }
        )

        with (EXPANDED_DIR / f"{dataset}_region_summary.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                region_rows.append(
                    {
                        "dataset": dataset,
                        "display_name": DISPLAY_NAMES[dataset],
                        "region": row["region"],
                        "count": row["count"],
                        "share_resolved_pct": row["share_resolved_pct"],
                    }
                )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "display_name",
                "total_items",
                "resolved_items",
                "resolved_coverage_pct",
                "western_share_resolved_pct",
                "top_country",
                "top_country_share_resolved_pct",
                "top3_country_share_resolved_pct",
                "top_region",
                "top_region_share_resolved_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with strict_compare_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "display_name",
                "total_items",
                "strict_resolved_items",
                "strict_resolved_coverage_pct",
                "expanded_resolved_items",
                "expanded_resolved_coverage_pct",
                "coverage_gain_points",
            ],
        )
        writer.writeheader()
        writer.writerows(strict_rows)

    with region_long_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "display_name", "region", "count", "share_resolved_pct"],
        )
        writer.writeheader()
        writer.writerows(region_rows)

    print(f"Wrote {summary_path}")
    print(f"Wrote {strict_compare_path}")
    print(f"Wrote {region_long_path}")


if __name__ == "__main__":
    build_cross_dataset_summary()
