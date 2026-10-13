from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEXICON = ROOT / "data" / "country_resolver_lexicons" / "country_lexicon.csv"
DEFAULT_WEIGHTS = {
    "country": 4,
    "demonym": 3,
    "city": 2,
}
TIER_RANK = {
    "none": 0,
    "medium": 1,
    "high": 2,
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("-", " ")
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S")):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return " ".join("".join(cleaned).split())


def alias_matches_text(alias_norm: str, text_norm: str, padded_text: str) -> bool:
    if all(ord(ch) < 128 for ch in alias_norm):
        return f" {alias_norm} " in padded_text
    return alias_norm in text_norm


def load_lexicon(path: Path, allowed_match_types: set[str]) -> dict[str, object]:
    ascii_lookup: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    unicode_lookup: list[tuple[str, list[dict[str, str]]]] = []
    unicode_tmp: dict[str, list[dict[str, str]]] = defaultdict(list)
    max_tokens = 1
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["match_type"] not in allowed_match_types:
                continue
            alias_norm = row["alias_norm"]
            if all(ord(ch) < 128 for ch in alias_norm):
                token_len = len(alias_norm.split())
                max_tokens = max(max_tokens, token_len)
                ascii_lookup[token_len][alias_norm].append(row)
            else:
                unicode_tmp[alias_norm].append(row)
    unicode_lookup = list(unicode_tmp.items())
    return {
        "ascii": ascii_lookup,
        "unicode": unicode_lookup,
        "max_tokens": max_tokens,
    }


def iter_caption_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def extract_matches(text: str, lexicon: dict[str, object]) -> list[dict[str, str]]:
    text_norm = normalize(text)
    matches: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    tokens = text_norm.split()

    ascii_lookup = lexicon["ascii"]
    max_tokens = lexicon["max_tokens"]
    for n in range(1, min(max_tokens, len(tokens)) + 1):
        candidates = ascii_lookup.get(n, {})
        if not candidates:
            continue
        for i in range(len(tokens) - n + 1):
            alias_norm = " ".join(tokens[i : i + n])
            if alias_norm not in candidates:
                continue
            for row in candidates[alias_norm]:
                key = (row["country_alpha2"], row["alias_norm"], row["match_type"])
                if key in seen:
                    continue
                matches.append(row)
                seen.add(key)

    for alias_norm, rows in lexicon["unicode"]:
        if alias_norm not in text_norm:
            continue
        for row in rows:
            key = (row["country_alpha2"], row["alias_norm"], row["match_type"])
            if key in seen:
                continue
            matches.append(row)
            seen.add(key)
    return matches


def strongest_tier(matches: list[dict[str, str]]) -> str:
    best = "none"
    for match in matches:
        tier = match.get("confidence_tier", "none")
        if TIER_RANK.get(tier, 0) > TIER_RANK.get(best, 0):
            best = tier
    return best


def choose_country(
    matches: list[dict[str, str]],
    match_type_weights: dict[str, int],
) -> tuple[str, str, Counter[str], Counter[str]]:
    if not matches:
        return "UNRESOLVED", "unresolved", Counter(), Counter()

    country_scores: Counter[str] = Counter()
    country_match_counts: Counter[str] = Counter()

    for match in matches:
        weight = match_type_weights.get(match["match_type"], 1)
        country = match["country_alpha2"]
        country_scores[country] += weight
        country_match_counts[country] += 1

    if len(country_scores) == 1:
        return next(iter(country_scores.keys())), "single_country", country_scores, country_match_counts

    top = country_scores.most_common()
    if len(top) >= 2 and top[0][1] > top[1][1]:
        return top[0][0], "weighted_majority_country", country_scores, country_match_counts

    return "AMBIGUOUS", "ambiguous", country_scores, country_match_counts


def parse_weight_string(raw: str) -> dict[str, int]:
    weights = dict(DEFAULT_WEIGHTS)
    if not raw.strip():
        return weights
    for chunk in raw.split(","):
        name, value = chunk.split("=", 1)
        weights[name.strip()] = int(value.strip())
    return weights


def run(
    input_csv: Path,
    text_column: str,
    id_column: str,
    output_csv: Path,
    summary_csv: Path,
    lexicon_csv: Path,
    allowed_match_types: set[str],
    match_type_weights: dict[str, int],
) -> None:
    lexicon = load_lexicon(lexicon_csv, allowed_match_types)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    country_counter = Counter()
    status_counter = Counter()
    tier_counter = Counter()
    dominant_match_type_counter = Counter()
    item_contains_type_counter = Counter()

    for row in iter_caption_rows(input_csv):
        item_id = row.get(id_column, "")
        text = row.get(text_column, "") or ""
        matches = extract_matches(text, lexicon)
        country_alpha2, resolution_status, country_scores, country_match_counts = choose_country(matches, match_type_weights)

        unique_aliases = sorted({m["alias"] for m in matches})
        unique_countries = sorted({m["country_alpha2"] for m in matches})
        unique_types = sorted({m["match_type"] for m in matches})
        score_repr = " | ".join(f"{country}:{score}" for country, score in country_scores.most_common())
        count_repr = " | ".join(f"{country}:{count}" for country, count in country_match_counts.most_common())
        strongest_type = max(
            (m["match_type"] for m in matches),
            key=lambda mt: match_type_weights.get(mt, 1),
            default="none",
        )
        confidence_tier = strongest_tier(matches)

        rows_out.append(
            {
                "item_id": item_id,
                "text": text,
                "matched_aliases": " | ".join(unique_aliases),
                "matched_types": " | ".join(unique_types),
                "candidate_countries": " | ".join(unique_countries),
                "candidate_country_scores": score_repr,
                "candidate_country_match_counts": count_repr,
                "resolved_country_alpha2": country_alpha2,
                "resolution_status": resolution_status,
                "resolution_confidence_tier": confidence_tier,
                "strongest_match_type": strongest_type,
                "match_count": str(len(matches)),
            }
        )
        status_counter[resolution_status] += 1
        tier_counter[confidence_tier] += 1
        dominant_match_type_counter[strongest_type] += 1
        for match_type in unique_types:
            item_contains_type_counter[match_type] += 1
        if country_alpha2 not in {"UNRESOLVED", "AMBIGUOUS"}:
            country_counter[country_alpha2] += 1

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "item_id",
                "text",
                "matched_aliases",
                "matched_types",
                "candidate_countries",
                "candidate_country_scores",
                "candidate_country_match_counts",
                "resolved_country_alpha2",
                "resolution_status",
                "resolution_confidence_tier",
                "strongest_match_type",
                "match_count",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    total = len(rows_out)
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerow({"metric": "total_items", "value": total})
        for key, value in sorted(status_counter.items()):
            writer.writerow({"metric": f"status::{key}", "value": value})
        for key, value in sorted(tier_counter.items()):
            writer.writerow({"metric": f"confidence::{key}", "value": value})
        for key, value in sorted(item_contains_type_counter.items()):
            writer.writerow({"metric": f"item_contains::{key}", "value": value})
        for key, value in sorted(dominant_match_type_counter.items()):
            writer.writerow({"metric": f"strongest_type::{key}", "value": value})
        for key, value in country_counter.most_common(25):
            writer.writerow({"metric": f"top_country::{key}", "value": value})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explicit non-LLM geography matcher for caption profiling.")
    parser.add_argument("--input-csv", required=True, help="Input CSV containing caption text.")
    parser.add_argument("--text-column", default="text", help="Column containing caption text.")
    parser.add_argument("--id-column", default="id", help="Column containing item ids.")
    parser.add_argument("--output-csv", required=True, help="Path for item-level output CSV.")
    parser.add_argument("--summary-csv", required=True, help="Path for summary CSV.")
    parser.add_argument("--lexicon-csv", default=str(DEFAULT_LEXICON), help="Country lexicon CSV.")
    parser.add_argument(
        "--allowed-match-types",
        default="country,demonym,city",
        help="Comma-separated match types to use from the lexicon.",
    )
    parser.add_argument(
        "--match-type-weights",
        default="country=4,demonym=3,city=2",
        help="Comma-separated weights, e.g. country=4,demonym=3,city=2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        input_csv=Path(args.input_csv),
        text_column=args.text_column,
        id_column=args.id_column,
        output_csv=Path(args.output_csv),
        summary_csv=Path(args.summary_csv),
        lexicon_csv=Path(args.lexicon_csv),
        allowed_match_types={item.strip() for item in args.allowed_match_types.split(",") if item.strip()},
        match_type_weights=parse_weight_string(args.match_type_weights),
    )


if __name__ == "__main__":
    main()
