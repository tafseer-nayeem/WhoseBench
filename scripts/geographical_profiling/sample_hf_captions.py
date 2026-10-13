from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream a small caption sample from a Hugging Face dataset into CSV.")
    parser.add_argument("--dataset-id", required=True, help="Hugging Face dataset id, e.g. wikimedia/wit_base")
    parser.add_argument("--split", default="train", help="Dataset split to stream")
    parser.add_argument("--limit", type=int, default=1000, help="Number of rows to keep")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    parser.add_argument(
        "--dataset-kind",
        choices=["wit_base", "wit_per_language", "simple_text", "cc12m_json", "redcaps_wds"],
        required=True,
        help="Light schema adapter for the dataset",
    )
    parser.add_argument(
        "--language-filter",
        default=None,
        help="Optional language filter. For wit_base this checks whether the requested language is present in wit_features.language.",
    )
    parser.add_argument(
        "--exclude-language",
        default=None,
        help="Optional language exclusion. For wit_base this skips rows containing the requested language in wit_features.language.",
    )
    return parser.parse_args()


def iter_stream(dataset_id: str, split: str):
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    for row in ds:
        yield row


def unique_nonempty(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def collect_wit_language_texts(wit: dict, j: int) -> tuple[str, str]:
    fields = [
        "caption_reference_description",
        "caption_alt_text_description",
        "caption_title_and_reference_description",
        "context_section_description",
        "context_page_description",
        "page_title",
        "section_title",
        "hierarchical_section_title",
    ]
    texts = []
    used_fields = []
    for field in fields:
        values = wit.get(field, []) or []
        if j < len(values) and values[j]:
            texts.append(str(values[j]))
            used_fields.append(field)
    texts = unique_nonempty(texts)
    return " || ".join(texts), " | ".join(used_fields)


def wit_rows(
    stream,
    limit: int,
    language_filter: str | None,
    exclude_language: str | None,
):
    kept = 0
    for idx, row in enumerate(stream):
        wit = row.get("wit_features", {})
        languages = wit.get("language", []) or []
        if language_filter and language_filter not in languages:
            continue
        if exclude_language and exclude_language in languages:
            continue

        texts = []
        sources = []
        for j, lang in enumerate(languages):
            if language_filter and lang != language_filter:
                continue
            if exclude_language and lang == exclude_language:
                continue
            text, text_source = collect_wit_language_texts(wit, j)
            if text:
                texts.append(text)
                sources.append(f"{lang}:{text_source}")
        texts = unique_nonempty(texts)
        if not texts:
            continue

        yield {
            "id": f"wit_{idx}",
            "text": " || ".join(texts),
            "dataset": "wit_base",
            "languages": " | ".join(sorted(set(languages))),
            "source_url": row.get("metadata_url", "") or row.get("image_url", ""),
            "text_source": " || ".join(sources),
            "image_url": row.get("image_url", ""),
        }
        kept += 1
        if kept >= limit:
            break


def wit_per_language_rows(
    stream,
    limit: int,
    language_filter: str | None,
    exclude_language: str | None,
):
    kept = 0
    for idx, row in enumerate(stream):
        wit = row.get("wit_features", {})
        languages = wit.get("language", []) or []
        n = len(languages)
        for j in range(n):
            lang = languages[j]
            if language_filter and lang != language_filter:
                continue
            if exclude_language and lang == exclude_language:
                continue

            text, text_source = collect_wit_language_texts(wit, j)

            if not text:
                continue

            yield {
                "id": f"witlang_{idx}_{j}",
                "text": text,
                "dataset": "wit_base",
                "languages": lang,
                "source_url": (wit.get("page_url", []) or [""] * n)[j] if j < len(wit.get("page_url", []) or []) else row.get("metadata_url", ""),
                "text_source": text_source,
                "image_url": row.get("image_url", ""),
            }
            kept += 1
            if kept >= limit:
                return


def simple_text_rows(stream, limit: int):
    kept = 0
    for idx, row in enumerate(stream):
        text = row.get("text") or row.get("caption") or ""
        if not text:
            continue
        yield {
            "id": f"row_{idx}",
            "text": str(text),
            "dataset": "generic",
            "languages": "",
            "source_url": row.get("url", "") or row.get("image_url", ""),
            "text_source": "text" if row.get("text") else "caption",
            "image_url": row.get("image_url", "") or row.get("url", ""),
        }
        kept += 1
        if kept >= limit:
            break


def cc12m_json_rows(stream, limit: int):
    kept = 0
    for idx, row in enumerate(stream):
        blob = row.get("json") or {}
        text = blob.get("caption") or blob.get("detailed_caption") or ""
        if not text:
            continue
        yield {
            "id": f"cc12m_{idx}",
            "text": str(text),
            "dataset": "cc12m",
            "languages": "en",
            "source_url": row.get("__url__", ""),
            "text_source": "json.caption" if blob.get("caption") else "json.detailed_caption",
            "image_url": "",
        }
        kept += 1
        if kept >= limit:
            break


def redcaps_wds_rows(stream, limit: int):
    kept = 0
    for idx, row in enumerate(stream):
        txt_keys = sorted([key for key in row.keys() if key.endswith(".txt")])
        json_keys = sorted([key for key in row.keys() if key.endswith(".json")])
        for j, txt_key in enumerate(txt_keys):
            text = row.get(txt_key) or ""
            if not text:
                continue
            json_blob = row.get(json_keys[j], {}) if j < len(json_keys) else {}
            source_url = json_blob.get("url", "") if isinstance(json_blob, dict) else ""
            yield {
                "id": f"redcaps_{idx}_{j}",
                "text": str(text),
                "dataset": "redcaps",
                "languages": "en",
                "source_url": source_url,
                "text_source": txt_key,
                "image_url": "",
            }
            kept += 1
            if kept >= limit:
                return


def main() -> None:
    args = parse_args()
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream = iter_stream(args.dataset_id, args.split)
    if args.dataset_kind == "wit_base":
        rows = wit_rows(stream, args.limit, args.language_filter, args.exclude_language)
    elif args.dataset_kind == "wit_per_language":
        rows = wit_per_language_rows(stream, args.limit, args.language_filter, args.exclude_language)
    elif args.dataset_kind == "cc12m_json":
        rows = cc12m_json_rows(stream, args.limit)
    elif args.dataset_kind == "redcaps_wds":
        rows = redcaps_wds_rows(stream, args.limit)
    else:
        rows = simple_text_rows(stream, args.limit)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "dataset", "languages", "source_url", "text_source", "image_url"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote sample captions to {output_path}")


if __name__ == "__main__":
    main()
