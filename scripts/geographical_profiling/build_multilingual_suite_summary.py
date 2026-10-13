from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "outputs" / "tables" / "country_region_expanded"
OUT_DIR = ROOT / "outputs" / "tables" / "multilingual_suite"

DATASETS = [
    ("wit_en_10k", "WIT-en", "english_control"),
    ("wit_fr_10k", "WIT-fr", "multilingual_control"),
    ("wit_de_10k", "WIT-de", "multilingual_control"),
    ("wit_es_10k", "WIT-es", "multilingual_control"),
    ("wit_ja_10k", "WIT-ja", "multilingual_control"),
]


def load_metric_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["metric"]: row["value"] for row in csv.DictReader(handle)}


def first_row(path: Path) -> dict[str, str] | None:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), None)


def copy_csv(src: Path, dst: Path) -> None:
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for key, display_name, group in DATASETS:
        for suffix in ["country_summary.csv", "region_summary.csv", "metrics_summary.csv"]:
            copy_csv(SOURCE_DIR / f"{key}_{suffix}", OUT_DIR / f"{key}_{suffix}")

        metrics = load_metric_csv(OUT_DIR / f"{key}_metrics_summary.csv")
        top_country = first_row(OUT_DIR / f"{key}_country_summary.csv") or {}
        top_region = first_row(OUT_DIR / f"{key}_region_summary.csv") or {}

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

    output = OUT_DIR / "multilingual_suite_summary.csv"
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
