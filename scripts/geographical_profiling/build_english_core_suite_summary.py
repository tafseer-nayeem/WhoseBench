from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = ROOT / "outputs" / "tables" / "english_core_suite"

DATASETS = [
    ("conceptual_captions_20k", "Conceptual Captions", "paper_core"),
    ("datacomp_1b_20k", "DataComp1B", "paper_core"),
    ("wit_english_50k", "WIT English (merged)", "aux_control"),
    ("redcaps_20k", "RedCaps", "social_web"),
    ("cc12m_20k", "CC12M", "literature_expansion"),
    ("coyo_700m_20k", "COYO-700M", "literature_expansion"),
]


def load_metric_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def first_row(path: Path) -> dict[str, str] | None:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), None)


def build() -> None:
    output = TABLE_DIR / "english_core_suite_summary.csv"
    rows: list[dict[str, str]] = []

    for key, display_name, group in DATASETS:
        metrics = load_metric_csv(TABLE_DIR / f"{key}_metrics_summary.csv")
        top_country = first_row(TABLE_DIR / f"{key}_country_summary.csv") or {}
        top_region = first_row(TABLE_DIR / f"{key}_region_summary.csv") or {}

        rows.append(
            {
                "dataset": key,
                "display_name": display_name,
                "benchmark_group": group,
                "total_items": metrics.get("total_items", ""),
                "resolved_items": metrics.get("resolved_items", ""),
                "resolved_coverage_pct": metrics.get("resolved_coverage_pct", ""),
                "western_share_resolved_pct": metrics.get("western_share_resolved_pct", ""),
                "top_country": top_country.get("country_name", ""),
                "top_country_share_resolved_pct": top_country.get("share_resolved_pct", ""),
                "top3_country_share_resolved_pct": metrics.get("top3_country_share_resolved_pct", ""),
                "top_region": top_region.get("region", ""),
                "top_region_share_resolved_pct": top_region.get("share_resolved_pct", ""),
            }
        )

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "display_name",
                "benchmark_group",
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
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    build()
