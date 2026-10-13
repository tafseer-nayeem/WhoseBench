from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from explicit_geo_match import load_lexicon, extract_matches, choose_country, parse_weight_string
from sample_hf_captions import iter_stream, wit_rows, simple_text_rows, cc12m_json_rows, redcaps_wds_rows
from summarize_geo_matches import country_to_region


def stream_rows(dataset_id: str, split: str, dataset_kind: str, language_filter: str | None):
    stream = iter_stream(dataset_id, split)
    if dataset_kind == "wit_base":
        yield from wit_rows(stream, limit=10**18, language_filter=language_filter, exclude_language=None)
    elif dataset_kind == "cc12m_json":
        yield from cc12m_json_rows(stream, limit=10**18)
    elif dataset_kind == "redcaps_wds":
        yield from redcaps_wds_rows(stream, limit=10**18)
    else:
        yield from simple_text_rows(stream, limit=10**18)


def dump_checkpoint(
    path: Path,
    processed: int,
    status_counter: Counter,
    country_counter: Counter,
    region_counter: Counter,
) -> None:
    payload = {
        "processed_rows": processed,
        "status_counter": dict(status_counter),
        "country_counter": dict(country_counter),
        "region_counter": dict(region_counter),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_outputs(
    country_csv: Path,
    region_csv: Path,
    metrics_csv: Path,
    processed: int,
    status_counter: Counter,
    country_counter: Counter,
    region_counter: Counter,
) -> None:
    country_csv.parent.mkdir(parents=True, exist_ok=True)
    total_resolved = sum(country_counter.values())

    with country_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["country_alpha2", "count", "share_pct"])
        writer.writeheader()
        for alpha2, count in country_counter.most_common():
            writer.writerow(
                {
                    "country_alpha2": alpha2,
                    "count": count,
                    "share_pct": f"{(100.0 * count / total_resolved) if total_resolved else 0.0:.6f}",
                }
            )

    with region_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["region", "count", "share_pct"])
        writer.writeheader()
        for region, count in region_counter.most_common():
            writer.writerow(
                {
                    "region": region,
                    "count": count,
                    "share_pct": f"{(100.0 * count / total_resolved) if total_resolved else 0.0:.6f}",
                }
            )

    western_regions = {"Europe", "North America", "Oceania"}
    western_count = sum(region_counter[r] for r in western_regions)
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "processed_rows", "value": processed})
        writer.writerow({"metric": "resolved_items", "value": total_resolved})
        writer.writerow({"metric": "resolved_coverage_pct", "value": f"{(100.0 * total_resolved / processed) if processed else 0.0:.6f}"})
        writer.writerow({"metric": "western_share_pct", "value": f"{(100.0 * western_count / total_resolved) if total_resolved else 0.0:.6f}"})
        for key, value in sorted(status_counter.items()):
            writer.writerow({"metric": f"status::{key}", "value": value})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Checkpointed full-corpus country counter for open datasets.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-kind", choices=["wit_base", "simple_text", "cc12m_json", "redcaps_wds"], required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--language-filter", default=None)
    parser.add_argument("--lexicon-csv", required=True)
    parser.add_argument("--country-csv", required=True)
    parser.add_argument("--region-csv", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--checkpoint-json", required=True)
    parser.add_argument("--checkpoint-every", type=int, default=10000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-first-n", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--match-type-weights", default="country=4,demonym=3,city=2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")

    lexicon = load_lexicon(Path(args.lexicon_csv), {"country", "demonym", "city"})
    weights = parse_weight_string(args.match_type_weights)

    processed = 0
    status_counter = Counter()
    country_counter = Counter()
    region_counter = Counter()

    for idx, row in enumerate(stream_rows(args.dataset_id, args.split, args.dataset_kind, args.language_filter)):
        if idx < args.skip_first_n:
            continue
        if idx % args.num_shards != args.shard_index:
            continue
        if args.max_rows is not None and processed >= args.max_rows:
            break

        text = row.get("text", "") or ""
        matches = extract_matches(text, lexicon)
        alpha2, status, _, _ = choose_country(matches, weights)
        processed += 1
        status_counter[status] += 1
        if alpha2 not in {"UNRESOLVED", "AMBIGUOUS"}:
            country_counter[alpha2] += 1
            region_counter[country_to_region(alpha2)] += 1

        if processed % args.checkpoint_every == 0:
            dump_checkpoint(Path(args.checkpoint_json), processed, status_counter, country_counter, region_counter)
            write_outputs(Path(args.country_csv), Path(args.region_csv), Path(args.metrics_csv), processed, status_counter, country_counter, region_counter)

    dump_checkpoint(Path(args.checkpoint_json), processed, status_counter, country_counter, region_counter)
    write_outputs(Path(args.country_csv), Path(args.region_csv), Path(args.metrics_csv), processed, status_counter, country_counter, region_counter)
    print(f"Processed {processed} rows")
    print(f"Shard {args.shard_index + 1}/{args.num_shards}")
    print(f"Wrote {args.country_csv}")
    print(f"Wrote {args.region_csv}")
    print(f"Wrote {args.metrics_csv}")
    print(f"Wrote {args.checkpoint_json}")


if __name__ == "__main__":
    main()
