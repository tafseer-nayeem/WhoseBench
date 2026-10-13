from __future__ import annotations

import argparse
import csv
from pathlib import Path


TARGET = {"US", "GB", "CA"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact bias summary emphasizing US/UK/Canada concentration.")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("country_csvs", nargs="+")
    return parser.parse_args()


def dataset_name_from_path(path: Path) -> str:
    name = path.name
    if name.endswith("_country_summary.csv"):
        return name[: -len("_country_summary.csv")]
    return path.stem


def main() -> None:
    args = parse_args()
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for country_csv in args.country_csvs:
        path = Path(country_csv)
        total_resolved = 0
        anglo_core = 0
        top_country = ""
        top_country_share = 0.0

        with path.open(newline="", encoding="utf-8") as handle:
            reader = list(csv.DictReader(handle))
        for i, row in enumerate(reader):
            count = int(row["count"])
            total_resolved += count
            if row["country_alpha2"] in TARGET:
                anglo_core += count
            if i == 0:
                top_country = row["country_alpha2"]
                top_country_share = float(row["share_resolved_pct"])

        rows.append(
            {
                "dataset_slice": dataset_name_from_path(path),
                "resolved_items": total_resolved,
                "us_uk_ca_count": anglo_core,
                "us_uk_ca_share_resolved_pct": f"{(100.0 * anglo_core / total_resolved) if total_resolved else 0.0:.4f}",
                "top_country": top_country,
                "top_country_share_resolved_pct": f"{top_country_share:.4f}",
            }
        )

    rows.sort(key=lambda r: r["dataset_slice"])
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset_slice",
                "resolved_items",
                "us_uk_ca_count",
                "us_uk_ca_share_resolved_pct",
                "top_country",
                "top_country_share_resolved_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
