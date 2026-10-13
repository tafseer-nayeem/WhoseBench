from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MULTI_DIR = ROOT / "outputs" / "tables" / "multilingual_suite"

SOURCE_KEYS = ["wit_fr_10k", "wit_de_10k", "wit_es_10k", "wit_ja_10k"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    country_counter = Counter()
    region_counter = Counter()
    total_items = 0
    total_resolved = 0

    for key in SOURCE_KEYS:
        metrics = {row["metric"]: row["value"] for row in load_csv(MULTI_DIR / f"{key}_metrics_summary.csv")}
        total_items += int(metrics.get("total_items", "0"))
        total_resolved += int(metrics.get("resolved_items", "0"))

        for row in load_csv(MULTI_DIR / f"{key}_country_summary.csv"):
            country_counter[(row["country_alpha2"], row["country_name"], row["region"])] += int(row["count"])
        for row in load_csv(MULTI_DIR / f"{key}_region_summary.csv"):
            region_counter[row["region"]] += int(row["count"])

    country_path = MULTI_DIR / "wit_multilingual_merged_country_summary.csv"
    region_path = MULTI_DIR / "wit_multilingual_merged_region_summary.csv"
    metrics_path = MULTI_DIR / "wit_multilingual_merged_metrics_summary.csv"

    with country_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["country_alpha2", "country_name", "region", "count", "share_resolved_pct"])
        writer.writeheader()
        for (alpha2, country_name, region), count in country_counter.most_common():
            writer.writerow(
                {
                    "country_alpha2": alpha2,
                    "country_name": country_name,
                    "region": region,
                    "count": count,
                    "share_resolved_pct": f"{(100.0 * count / total_resolved) if total_resolved else 0.0:.4f}",
                }
            )

    with region_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["region", "count", "share_resolved_pct"])
        writer.writeheader()
        for region, count in region_counter.most_common():
            writer.writerow(
                {
                    "region": region,
                    "count": count,
                    "share_resolved_pct": f"{(100.0 * count / total_resolved) if total_resolved else 0.0:.4f}",
                }
            )

    western_regions = {"Europe", "North America", "Oceania"}
    western_count = sum(count for region, count in region_counter.items() if region in western_regions)
    top_country_count = country_counter.most_common(1)[0][1] if country_counter else 0
    top3_country_count = sum(count for _, count in country_counter.most_common(3))

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "total_items", "value": total_items})
        writer.writerow({"metric": "resolved_items", "value": total_resolved})
        writer.writerow({"metric": "resolved_coverage_pct", "value": f"{(100.0 * total_resolved / total_items) if total_items else 0.0:.4f}"})
        writer.writerow({"metric": "western_share_resolved_pct", "value": f"{(100.0 * western_count / total_resolved) if total_resolved else 0.0:.4f}"})
        writer.writerow({"metric": "top_country_share_resolved_pct", "value": f"{(100.0 * top_country_count / total_resolved) if total_resolved else 0.0:.4f}"})
        writer.writerow({"metric": "top3_country_share_resolved_pct", "value": f"{(100.0 * top3_country_count / total_resolved) if total_resolved else 0.0:.4f}"})
        for region, count in region_counter.most_common():
            writer.writerow({"metric": f"region::{region}", "value": count})

    print(f"Wrote {country_path}")
    print(f"Wrote {region_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
