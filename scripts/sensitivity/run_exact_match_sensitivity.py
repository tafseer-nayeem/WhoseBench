from __future__ import annotations

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[2]
INVENTORY = ROOT / "data" / "convention_detector_inventory"
OUTPUT_DIR = ROOT / "results" / "evaluation_sensitivity"


ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
MULTISPACE = re.compile(r"\s+")
MONTHS = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

UNIT_VARIANTS = {
    "meter": "metre",
    "meters": "metres",
    "kilometer": "kilometre",
    "kilometers": "kilometres",
    "liter": "litre",
    "liters": "litres",
    "milliliter": "millilitre",
    "milliliters": "millilitres",
}

SAFE_SPELLING_SUFFIX_PAIRS = (
    ("or", "our"),
    ("ors", "ours"),
    ("ize", "ise"),
    ("izes", "ises"),
    ("ized", "ised"),
    ("izing", "ising"),
    ("ization", "isation"),
    ("izations", "isations"),
    ("izer", "iser"),
    ("izers", "isers"),
    ("yze", "yse"),
    ("yzes", "yses"),
    ("yzed", "ysed"),
    ("yzing", "ysing"),
    ("er", "re"),
    ("ers", "res"),
    ("ense", "ence"),
    ("enses", "ences"),
    ("eled", "elled"),
    ("eling", "elling"),
    ("eler", "eller"),
    ("elers", "ellers"),
    ("og", "ogue"),
    ("ogs", "ogues"),
    ("gram", "gramme"),
    ("grams", "grammes"),
)


@dataclass(frozen=True)
class AnswerItem:
    benchmark: str
    split: str
    item_id: str
    question: str
    references: tuple[str, ...]
    source: str


@dataclass(frozen=True)
class VariantRecord:
    benchmark: str
    split: str
    item_id: str
    source: str
    convention_type: str
    original_answer: str
    variant_answer: str
    accepted_by_exact_match: bool
    accepted_by_loose_normalized_match: bool
    benchmark_scorer_family: str
    accepted_by_benchmark_scorer: str
    benchmark_scorer_note: str
    num_references: int
    references_joined: str


def normalize_exact_match(text: str) -> str:
    """SQuAD/DROP-style normalization: lowercase, strip punctuation/articles/extra spaces."""
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = ARTICLES.sub(" ", text)
    text = MULTISPACE.sub(" ", text).strip()
    return text


def normalize_loose(text: str) -> str:
    """A slightly more permissive normalizer, still not convention-aware."""
    text = normalize_exact_match(text)
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s*-\s*", "-", text)
    return text


def exact_match(candidate: str, references: tuple[str, ...]) -> bool:
    c = normalize_exact_match(candidate)
    return any(c == normalize_exact_match(ref) for ref in references)


def loose_match(candidate: str, references: tuple[str, ...]) -> bool:
    c = normalize_loose(candidate)
    return any(c == normalize_loose(ref) for ref in references)


def token_f1(candidate: str, reference: str) -> float:
    candidate_tokens = normalize_exact_match(candidate).split()
    reference_tokens = normalize_exact_match(reference).split()
    if not candidate_tokens or not reference_tokens:
        return float(candidate_tokens == reference_tokens)
    candidate_counts = Counter(candidate_tokens)
    reference_counts = Counter(reference_tokens)
    common = sum((candidate_counts & reference_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(candidate_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_number(text: str) -> str | None:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def benchmark_scorer_result(item: AnswerItem, variant: str) -> tuple[str, str, str]:
    if item.benchmark == "DROP":
        accepted = exact_match(variant, item.references)
        max_f1 = max((token_f1(variant, ref) for ref in item.references), default=0.0)
        return (
            "DROP_official_style_EM_F1",
            str(bool(accepted or max_f1 >= 0.999999)),
            f"max_token_f1={max_f1:.3f}",
        )
    if item.benchmark == "GSM8K":
        candidate_number = extract_number(variant)
        ref_numbers = [extract_number(ref) for ref in item.references]
        accepted = candidate_number is not None and candidate_number in ref_numbers
        return ("GSM8K_final_number", str(bool(accepted)), f"candidate_number={candidate_number}")
    if item.source == "correct_option_text":
        return (
            "MCQ_official_label_not_answer_text",
            "not_applicable",
            "Official MCQ scoring uses the gold label; answer-text variants are tested only under text-normalized scoring.",
        )
    return (
        "standard_exact_match_reference",
        str(exact_match(variant, item.references)),
        "No separate official scorer available in artifact; standard EM is reported.",
    )


def preserve_case(src: str, dst: str) -> str:
    if src.isupper():
        return dst.upper()
    if src[:1].isupper():
        return dst[:1].upper() + dst[1:]
    return dst


def load_spelling_map(max_pairs: int | None = None, include_vocabulary: bool = False) -> dict[str, str]:
    path = INVENTORY / "us_uk_variations_categorized.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    mapping: dict[str, str] = {}
    for row in rows:
        if not include_vocabulary and row.get("category") != "Orthographic/Spelling":
            continue
        us = str(row.get("us", "")).strip()
        uk = str(row.get("uk", "")).strip()
        if us and uk and us.lower() != uk.lower() and is_safe_orthographic_pair(us, uk):
            mapping[us.lower()] = uk.lower()
        if max_pairs and len(mapping) >= max_pairs:
            break
    return mapping


def is_safe_orthographic_pair(us: str, uk: str) -> bool:
    """Keep high-precision spelling alternations and drop noisy lexical/proper-name pairs."""
    us_l = us.lower().strip()
    uk_l = uk.lower().strip()
    if not us_l.isalpha() or not uk_l.isalpha():
        return False
    for us_suffix, uk_suffix in SAFE_SPELLING_SUFFIX_PAIRS:
        if us_l.endswith(us_suffix) and uk_l.endswith(uk_suffix):
            return us_l[: -len(us_suffix)] == uk_l[: -len(uk_suffix)]
    return False


def compile_spelling_pattern(spelling_map: dict[str, str]) -> re.Pattern[str] | None:
    if not spelling_map:
        return None
    words = sorted(spelling_map, key=len, reverse=True)
    return re.compile(r"\b(" + "|".join(re.escape(word) for word in words) + r")\b", flags=re.IGNORECASE)


def spelling_variants(text: str, spelling_map: dict[str, str], spelling_pattern: re.Pattern[str] | None) -> list[str]:
    if spelling_pattern is None:
        return []
    if not spelling_pattern.search(text):
        return []

    def repl(match: re.Match[str]) -> str:
        src = match.group(0)
        return preserve_case(src, spelling_map[src.lower()])

    variant = spelling_pattern.sub(repl, text)
    return [variant] if variant != text else []


def unit_spelling_variants(text: str) -> list[str]:
    if not UNIT_VARIANTS:
        return []
    pattern = re.compile(
        r"\b(" + "|".join(re.escape(src) for src in sorted(UNIT_VARIANTS, key=len, reverse=True)) + r")\b",
        flags=re.IGNORECASE,
    )
    if not pattern.search(text):
        return []

    def repl(match: re.Match[str]) -> str:
        src = match.group(0)
        return preserve_case(src, UNIT_VARIANTS[src.lower()])

    variant = pattern.sub(repl, text)
    return [variant] if variant != text else []


def written_date_variants(text: str) -> list[str]:
    variants = set()

    # April 21, 2025 -> 21 April 2025. This keeps the same date and avoids numeric ambiguity.
    month_names = "|".join(sorted(MONTHS, key=len, reverse=True))
    pattern = re.compile(
        rf"\b({month_names})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,\s*(\d{{4}}))?\b",
        flags=re.IGNORECASE,
    )

    def repl(match: re.Match[str]) -> str:
        month = match.group(1)
        day = match.group(2)
        year = match.group(3)
        if year:
            return f"{int(day)} {preserve_case(month, month.lower())} {year}"
        return f"{int(day)} {preserve_case(month, month.lower())}"

    if pattern.search(text):
        variants.add(pattern.sub(repl, text))
    return sorted(v for v in variants if v != text)


def numeric_date_variants(text: str) -> list[str]:
    pattern = re.compile(r"\b(0?[1-9]|1[0-2])/(0?[1-9]|[12]\d|3[01])/(\d{4})\b")
    matches = list(pattern.finditer(text))
    if not matches:
        return []
    pieces = []
    cursor = 0
    changed = False
    for match in matches:
        month = int(match.group(1))
        day = int(match.group(2))
        year = match.group(3)
        pieces.append(text[cursor : match.start()])
        if day <= 12:
            pieces.append(match.group(0))
        else:
            pieces.append(f"{day:02d}/{month:02d}/{year}")
            changed = True
        cursor = match.end()
    pieces.append(text[cursor:])
    variant = "".join(pieces)
    return [variant] if changed and variant != text else []


def time_variants(text: str) -> list[str]:
    pattern = re.compile(r"\b(1[0-2]|0?[1-9])(?::([0-5]\d))?\s*([ap])\.?m\.?\b", flags=re.IGNORECASE)
    if not pattern.search(text):
        return []

    def repl(match: re.Match[str]) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        meridiem = match.group(3).lower()
        if meridiem == "a":
            hour24 = 0 if hour == 12 else hour
        else:
            hour24 = 12 if hour == 12 else hour + 12
        return f"{hour24:02d}:{minute:02d}"

    variant = pattern.sub(repl, text)
    return [variant] if variant != text else []


def build_variants(
    text: str,
    spelling_map: dict[str, str],
    spelling_pattern: re.Pattern[str] | None,
) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for v in spelling_variants(text, spelling_map, spelling_pattern):
        out.append(("spelling", v))
    for v in unit_spelling_variants(text):
        out.append(("unit_spelling", v))
    for v in written_date_variants(text):
        out.append(("written_date", v))
    for v in numeric_date_variants(text):
        out.append(("numeric_date_unambiguous", v))
    for v in time_variants(text):
        out.append(("time_12h_to_24h", v))
    # Deterministic order and deduplication.
    seen = set()
    deduped = []
    for kind, variant in out:
        key = (kind, variant)
        if key not in seen:
            seen.add(key)
            deduped.append((kind, variant))
    return deduped


def clean_refs(values) -> tuple[str, ...]:
    refs: list[str] = []
    if isinstance(values, str):
        refs.append(values)
    elif isinstance(values, (list, tuple)):
        for value in values:
            refs.extend(clean_refs(value))
    elif isinstance(values, dict):
        for value in values.values():
            refs.extend(clean_refs(value))
    elif values is not None:
        refs.append(str(values))
    refs = [MULTISPACE.sub(" ", str(ref).strip()) for ref in refs if str(ref).strip()]
    # Stable unique order.
    return tuple(dict.fromkeys(refs))


def iter_answer_items(benchmarks: list[str], max_examples_per_benchmark: int | None) -> list[AnswerItem]:
    items: list[AnswerItem] = []

    def add(item: AnswerItem) -> None:
        if max_examples_per_benchmark is None:
            items.append(item)
            return
        count = sum(1 for existing in items if existing.benchmark == item.benchmark)
        if count < max_examples_per_benchmark:
            items.append(item)

    if "drop" in benchmarks:
        ds = load_dataset("ucinlp/drop", split="validation")
        for idx, row in enumerate(ds):
            refs = clean_refs((row.get("answers_spans") or {}).get("spans", []))
            if refs:
                add(
                    AnswerItem(
                        benchmark="DROP",
                        split="validation",
                        item_id=str(row.get("query_id") or idx),
                        question=str(row.get("question", "")),
                        references=refs,
                        source="gold_answer_spans",
                    )
                )

    if "truthfulqa" in benchmarks:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
        for idx, row in enumerate(ds):
            refs = clean_refs([row.get("best_answer", "")] + list(row.get("correct_answers") or []))
            if refs:
                add(
                    AnswerItem(
                        benchmark="TruthfulQA",
                        split="validation",
                        item_id=f"truthfulqa_{idx}",
                        question=str(row.get("question", "")),
                        references=refs,
                        source="correct_answer_aliases",
                    )
                )

    if "gsm8k" in benchmarks:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        for idx, row in enumerate(ds):
            answer = str(row.get("answer", ""))
            final = answer.split("####")[-1].strip() if "####" in answer else answer.strip()
            refs = clean_refs([final])
            if refs:
                add(
                    AnswerItem(
                        benchmark="GSM8K",
                        split="test",
                        item_id=f"gsm8k_{idx}",
                        question=str(row.get("question", "")),
                        references=refs,
                        source="final_answer",
                    )
                )

    if "arc_challenge" in benchmarks:
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
        for idx, row in enumerate(ds):
            choices = row.get("choices") or {}
            labels = list(choices.get("label") or [])
            texts = list(choices.get("text") or [])
            answer_key = str(row.get("answerKey", ""))
            refs = []
            for label, text in zip(labels, texts):
                if str(label) == answer_key:
                    refs.append(str(text))
            if refs:
                add(
                    AnswerItem(
                        benchmark="ARC-Challenge",
                        split="validation",
                        item_id=str(row.get("id") or idx),
                        question=str(row.get("question", "")),
                        references=clean_refs(refs),
                        source="correct_option_text",
                    )
                )

    if "arc_easy" in benchmarks:
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        for idx, row in enumerate(ds):
            choices = row.get("choices") or {}
            labels = list(choices.get("label") or [])
            texts = list(choices.get("text") or [])
            answer_key = str(row.get("answerKey", ""))
            refs = []
            for label, text in zip(labels, texts):
                if str(label) == answer_key:
                    refs.append(str(text))
            if refs:
                add(
                    AnswerItem(
                        benchmark="ARC-Easy",
                        split="validation",
                        item_id=str(row.get("id") or idx),
                        question=str(row.get("question", "")),
                        references=clean_refs(refs),
                        source="correct_option_text",
                    )
                )

    if "openbookqa" in benchmarks:
        ds = load_dataset("allenai/openbookqa", "main", split="validation")
        for idx, row in enumerate(ds):
            choices = row.get("choices") or {}
            labels = list(choices.get("label") or [])
            texts = list(choices.get("text") or [])
            answer_key = str(row.get("answerKey", ""))
            refs = []
            for label, text in zip(labels, texts):
                if str(label) == answer_key:
                    refs.append(str(text))
            if refs:
                add(
                    AnswerItem(
                        benchmark="OpenBookQA",
                        split="validation",
                        item_id=str(row.get("id") or idx),
                        question=str(row.get("question_stem", "")),
                        references=clean_refs(refs),
                        source="correct_option_text",
                    )
                )

    if "mmlu" in benchmarks:
        ds = load_dataset("cais/mmlu", "all", split="validation")
        for idx, row in enumerate(ds):
            choices = list(row.get("choices") or [])
            answer_idx = row.get("answer")
            refs = []
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                refs.append(str(choices[answer_idx]))
            if refs:
                add(
                    AnswerItem(
                        benchmark="MMLU",
                        split="validation",
                        item_id=f"mmlu_validation_{idx}",
                        question=str(row.get("question", "")),
                        references=clean_refs(refs),
                        source="correct_option_text",
                    )
                )

    if "mmlu_pro" in benchmarks:
        for split in ["validation", "test"]:
            try:
                ds = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
            except Exception:
                continue
            for idx, row in enumerate(ds):
                choices = list(row.get("options") or [])
                answer_idx = row.get("answer_index")
                refs = []
                if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices):
                    refs.append(str(choices[answer_idx]))
                if refs:
                    add(
                        AnswerItem(
                            benchmark="MMLU-Pro",
                            split=split,
                            item_id=str(row.get("question_id") or f"mmlu_pro_{split}_{idx}"),
                            question=str(row.get("question", "")),
                            references=clean_refs(refs),
                            source="correct_option_text",
                        )
                    )

    if "hellaswag" in benchmarks:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        for idx, row in enumerate(ds):
            endings = list(row.get("endings") or [])
            label = str(row.get("label", ""))
            refs = []
            if label.isdigit() and int(label) < len(endings):
                refs.append(str(endings[int(label)]))
            if refs:
                add(
                    AnswerItem(
                        benchmark="HellaSwag",
                        split="validation",
                        item_id=str(row.get("ind") or idx),
                        question=str(row.get("ctx", "")),
                        references=clean_refs(refs),
                        source="correct_option_text",
                    )
                )

    if "boolq" in benchmarks:
        ds = load_dataset("google/boolq", split="validation")
        for idx, row in enumerate(ds):
            refs = clean_refs(["yes" if row.get("answer") else "no"])
            add(
                AnswerItem(
                    benchmark="BoolQ",
                    split="validation",
                    item_id=f"boolq_{idx}",
                    question=str(row.get("question", "")),
                    references=refs,
                    source="boolean_answer_text",
                )
            )

    if "bbh_date" in benchmarks:
        ds = load_dataset("lukaemon/bbh", "date_understanding", split="test")
        for idx, row in enumerate(ds):
            refs = correct_bbh_option_text(str(row.get("input", "")), str(row.get("target", "")))
            if refs:
                add(
                    AnswerItem(
                        benchmark="BBH-date-understanding",
                        split="test",
                        item_id=f"bbh_date_{idx}",
                        question=str(row.get("input", "")),
                        references=clean_refs(refs),
                        source="correct_option_text",
                    )
                )

    if "bbh_temporal" in benchmarks:
        ds = load_dataset("lukaemon/bbh", "temporal_sequences", split="test")
        for idx, row in enumerate(ds):
            refs = correct_bbh_option_text(str(row.get("input", "")), str(row.get("target", "")))
            if refs:
                add(
                    AnswerItem(
                        benchmark="BBH-temporal-sequences",
                        split="test",
                        item_id=f"bbh_temporal_{idx}",
                        question=str(row.get("input", "")),
                        references=clean_refs(refs),
                        source="correct_option_text",
                    )
                )

    return items


def correct_bbh_option_text(prompt: str, target: str) -> tuple[str, ...]:
    match = re.search(r"\(([A-Z])\)", target)
    if not match:
        return ()
    gold = match.group(1)
    option_pattern = re.compile(rf"\({gold}\)\s*([^\n]+)")
    option_match = option_pattern.search(prompt)
    if option_match:
        return (option_match.group(1).strip(),)
    return ()


def summarize(records: list[VariantRecord]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[VariantRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.benchmark, record.convention_type)].append(record)
        grouped[(record.benchmark, "ALL")].append(record)
        grouped[("ALL", record.convention_type)].append(record)
        grouped[("ALL", "ALL")].append(record)

    rows = []
    for (benchmark, convention_type), recs in sorted(grouped.items()):
        rows.append(
            {
                "benchmark": benchmark,
                "convention_type": convention_type,
                "num_variant_records": len(recs),
                "num_source_items": len({r.item_id for r in recs}),
                "exact_match_accept_rate": mean(r.accepted_by_exact_match for r in recs),
                "exact_match_rejection_rate": 1.0 - mean(r.accepted_by_exact_match for r in recs),
                "loose_match_accept_rate": mean(r.accepted_by_loose_normalized_match for r in recs),
                "loose_match_rejection_rate": 1.0 - mean(r.accepted_by_loose_normalized_match for r in recs),
                "benchmark_scorer_applicable_records": sum(r.accepted_by_benchmark_scorer != "not_applicable" for r in recs),
                "benchmark_scorer_rejection_rate_when_applicable": (
                    mean(r.accepted_by_benchmark_scorer == "False" for r in recs if r.accepted_by_benchmark_scorer != "not_applicable")
                    if any(r.accepted_by_benchmark_scorer != "not_applicable" for r in recs)
                    else ""
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, object]], records: list[VariantRecord], skipped: Counter[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top = [row for row in summary_rows if row["benchmark"] != "ALL" and row["convention_type"] == "ALL"]
    overall = next((row for row in summary_rows if row["benchmark"] == "ALL" and row["convention_type"] == "ALL"), None)
    lines = [
        "# Exact-Match Evaluation Sensitivity",
        "",
        "This audit asks whether user-valid answer variants would be accepted by a standard exact-match",
        "normalizer when the benchmark reference set contains only the original convention form.",
        "",
        "The experiment does not measure model behavior. It measures answer-normalization sensitivity:",
        "if a model or user produced the valid variant, would a conventional exact-match scorer accept it?",
        "For benchmarks with distinct official scoring semantics, the table also records whether the",
        "benchmark-level scorer is applicable to answer text variants.",
        "",
    ]
    if overall:
        official_rej = overall["benchmark_scorer_rejection_rate_when_applicable"]
        official_rej_text = "" if official_rej == "" else f"{official_rej:.3f}"
        lines.extend(
            [
                "## Headline",
                "",
                f"- Variant records: {overall['num_variant_records']}",
                f"- Source items: {overall['num_source_items']}",
                f"- Exact-match rejection rate: {overall['exact_match_rejection_rate']:.3f}",
                f"- Loose normalized rejection rate: {overall['loose_match_rejection_rate']:.3f}",
                f"- Benchmark-scorer applicable records: {overall['benchmark_scorer_applicable_records']}",
                f"- Benchmark-scorer rejection rate when applicable: {official_rej_text}",
                "",
            ]
        )
    lines.extend(["## Benchmark Summary", ""])
    lines.append("| Benchmark | Variants | Source items | Exact rejection | Official/applicable records | Official rejection when applicable |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in sorted(top, key=lambda r: str(r["benchmark"])):
        official_rej = row["benchmark_scorer_rejection_rate_when_applicable"]
        official_rej_text = "" if official_rej == "" else f"{official_rej:.3f}"
        lines.append(
            f"| {row['benchmark']} | {row['num_variant_records']} | {row['num_source_items']} | "
            f"{row['exact_match_rejection_rate']:.3f} | {row['benchmark_scorer_applicable_records']} | "
            f"{official_rej_text} |"
        )
    rejected_examples = [
        record
        for record in records
        if not record.accepted_by_exact_match
        and record.accepted_by_benchmark_scorer in {"False", "not_applicable"}
    ]
    lines.extend(["", "## Example Rejected Variants", ""])
    for record in rejected_examples[:12]:
        lines.append(
            f"- **{record.benchmark}** `{record.convention_type}`: `{record.original_answer}` -> "
            f"`{record.variant_answer}`; references: `{record.references_joined}`"
        )
    lines.extend(["", "## Skips", ""])
    if skipped:
        for reason, count in skipped.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- None")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmarks",
        default="drop,truthfulqa,gsm8k,arc_challenge,arc_easy,openbookqa,mmlu,mmlu_pro,hellaswag,boolq,bbh_date,bbh_temporal",
        help=(
            "Comma-separated benchmarks: drop,truthfulqa,gsm8k,arc_challenge,arc_easy,"
            "openbookqa,mmlu,mmlu_pro,hellaswag,boolq,bbh_date,bbh_temporal"
        ),
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max-examples-per-benchmark", type=int, default=None)
    parser.add_argument("--max-spelling-pairs", type=int, default=None)
    parser.add_argument(
        "--include-vocabulary-variants",
        action="store_true",
        help="Also include US/UK vocabulary alternatives. Off by default to keep the variant set conservative.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    benchmarks = [part.strip() for part in args.benchmarks.split(",") if part.strip()]
    spelling_map = load_spelling_map(args.max_spelling_pairs, include_vocabulary=args.include_vocabulary_variants)
    spelling_pattern = compile_spelling_pattern(spelling_map)
    items = iter_answer_items(benchmarks, args.max_examples_per_benchmark)

    records: list[VariantRecord] = []
    skipped: Counter[str] = Counter()
    for item in items:
        item_has_variant = False
        for answer in item.references:
            variants = build_variants(answer, spelling_map, spelling_pattern)
            if not variants:
                continue
            item_has_variant = True
            for convention_type, variant in variants:
                scorer_family, scorer_accept, scorer_note = benchmark_scorer_result(item, variant)
                records.append(
                    VariantRecord(
                        benchmark=item.benchmark,
                        split=item.split,
                        item_id=item.item_id,
                        source=item.source,
                        convention_type=convention_type,
                        original_answer=answer,
                        variant_answer=variant,
                        accepted_by_exact_match=exact_match(variant, item.references),
                        accepted_by_loose_normalized_match=loose_match(variant, item.references),
                        benchmark_scorer_family=scorer_family,
                        accepted_by_benchmark_scorer=scorer_accept,
                        benchmark_scorer_note=scorer_note,
                        num_references=len(item.references),
                        references_joined=" || ".join(item.references[:8]),
                    )
                )
        if not item_has_variant:
            skipped["no_safe_answer_variant"] += 1

    record_rows = [record.__dict__ for record in records]
    summary_rows = summarize(records) if records else []
    write_csv(output_dir / "exact_match_variant_records.csv", record_rows)
    write_csv(output_dir / "exact_match_variant_summary.csv", summary_rows)
    write_csv(output_dir / "exact_match_skip_summary.csv", [{"reason": k, "count": v} for k, v in skipped.items()])
    write_markdown(output_dir / "EXACT_MATCH_SENSITIVITY_FINDINGS.md", summary_rows, records, skipped)

    print(
        {
            "items_scanned": len(items),
            "variant_records": len(records),
            "summary_rows": len(summary_rows),
            "output_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    main()
