from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import pycountry


CONTINENT_MAP = {
    "AF": "Africa",
    "AN": "Antarctica",
    "AS": "Asia",
    "EU": "Europe",
    "NA": "North America",
    "OC": "Oceania",
    "SA": "South America",
}
WESTERN_REGIONS = {"Europe", "North America", "Oceania"}


def country_to_region(alpha2: str) -> str:
    try:
        import pycountry_convert as pc

        continent_code = pc.country_alpha2_to_continent_code(alpha2)
        return CONTINENT_MAP.get(continent_code, "Unknown")
    except Exception:
        return "Unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize resolved geography matches into country, region, and metrics tables."
    )
    parser.add_argument("--input-csv", required=True, help="Item-level explicit_geo_match output")
    parser.add_argument("--country-csv", required=True, help="Output country summary CSV")
    parser.add_argument("--region-csv", help="Output region summary CSV")
    parser.add_argument("--continent-csv", help="Backward-compatible alias for --region-csv")
    parser.add_argument("--metrics-csv", required=True, help="Output aggregate metrics CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv)
    country_path = Path(args.country_csv)
    region_path = Path(args.region_csv or args.continent_csv)
    metrics_path = Path(args.metrics_csv)
    country_path.parent.mkdir(parents=True, exist_ok=True)
    region_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    country_counter = Counter()
    region_counter = Counter()
    status_counter = Counter()
    tier_counter = Counter()
    strongest_type_counter = Counter()
    total_items = 0
    total_resolved = 0

    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total_items += 1
            status = row.get("resolution_status", "unknown")
            tier = row.get("resolution_confidence_tier", "none")
            strongest_type = row.get("strongest_match_type", "none")
            status_counter[status] += 1
            tier_counter[tier] += 1
            strongest_type_counter[strongest_type] += 1

            alpha2 = row["resolved_country_alpha2"]
            if alpha2 in {"UNRESOLVED", "AMBIGUOUS"}:
                continue
            total_resolved += 1
            country_counter[alpha2] += 1
            region_counter[country_to_region(alpha2)] += 1

    with country_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["country_alpha2", "country_name", "region", "count", "share_resolved_pct"],
        )
        writer.writeheader()
        for alpha2, count in country_counter.most_common():
            country = pycountry.countries.get(alpha_2=alpha2)
            writer.writerow(
                {
                    "country_alpha2": alpha2,
                    "country_name": getattr(country, "name", alpha2),
                    "region": country_to_region(alpha2),
                    "count": count,
                    "share_resolved_pct": f"{(100.0 * count / total_resolved) if total_resolved else 0.0:.4f}",
                }
            )

    with region_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["region", "count", "share_resolved_pct"],
        )
        writer.writeheader()
        for region, count in region_counter.most_common():
            writer.writerow(
                {
                    "region": region,
                    "count": count,
                    "share_resolved_pct": f"{(100.0 * count / total_resolved) if total_resolved else 0.0:.4f}",
                }
            )

    western_count = sum(count for region, count in region_counter.items() if region in WESTERN_REGIONS)
    top_country_count = country_counter.most_common(1)[0][1] if country_counter else 0
    top3_country_count = sum(count for _, count in country_counter.most_common(3))

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "total_items", "value": total_items})
        writer.writerow({"metric": "resolved_items", "value": total_resolved})
        writer.writerow(
            {
                "metric": "resolved_coverage_pct",
                "value": f"{(100.0 * total_resolved / total_items) if total_items else 0.0:.4f}",
            }
        )
        writer.writerow(
            {
                "metric": "western_share_resolved_pct",
                "value": f"{(100.0 * western_count / total_resolved) if total_resolved else 0.0:.4f}",
            }
        )
        writer.writerow(
            {
                "metric": "top_country_share_resolved_pct",
                "value": f"{(100.0 * top_country_count / total_resolved) if total_resolved else 0.0:.4f}",
            }
        )
        writer.writerow(
            {
                "metric": "top3_country_share_resolved_pct",
                "value": f"{(100.0 * top3_country_count / total_resolved) if total_resolved else 0.0:.4f}",
            }
        )
        for key, value in sorted(status_counter.items()):
            writer.writerow({"metric": f"status::{key}", "value": value})
        for key, value in sorted(tier_counter.items()):
            writer.writerow({"metric": f"confidence::{key}", "value": value})
        for key, value in sorted(strongest_type_counter.items()):
            writer.writerow({"metric": f"strongest_type::{key}", "value": value})
        for key, value in region_counter.most_common():
            writer.writerow({"metric": f"region::{key}", "value": value})

    print(f"Resolved items: {total_resolved}")
    print(f"Wrote {country_path}")
    print(f"Wrote {region_path}")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
