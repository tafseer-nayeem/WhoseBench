from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATASETS = DATA_DIR / "datasets"
DATASETS_HF = DATA_DIR / "datasets_hf"
RESOURCES = DATA_DIR / "resources"
OUTPUT_DIR = ROOT / "outputs" / "analysis"


MONTH_PATTERN = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"


CONVENTION_PATTERNS = {
    "american_date_instruction": re.compile(r"\bMM/DD/YYYY\b", re.IGNORECASE),
    "american_written_date": re.compile(
        rf"\b{MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*\d{{4}})?\b",
        re.IGNORECASE,
    ),
    "american_time_12h": re.compile(r"\b\d{1,2}(?::\d{2})?\s?(?:am|pm)\b", re.IGNORECASE),
    "american_customary_unit": re.compile(
        r"\b(?:lb|lbs|pound|pounds|oz|ounce|ounces|inch|inches|foot|feet|yard|yards|mile|miles|gallon|gallons|quart|quarts|fahrenheit)\b|°\s?f\b",
        re.IGNORECASE,
    ),
    "other_date_instruction": re.compile(r"\bDD/MM/YYYY\b|\bYYYY-MM-DD\b", re.IGNORECASE),
    "other_written_date": re.compile(
        rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTH_PATTERN}(?:\s+\d{{4}})?\b",
        re.IGNORECASE,
    ),
    "other_time_24h": re.compile(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b(?!\s?(?:am|pm))", re.IGNORECASE),
    "other_metric_unit": re.compile(
        r"\b(?:kg|kilogram|kilograms|gram|grams|cm|meter|meters|metre|metres|km|kilometer|kilometers|kilometre|kilometres|liter|liters|litre|litres|ml|celsius|stone|stones)\b|°\s?c\b",
        re.IGNORECASE,
    ),
}

AMERICAN_PATTERN_KEYS = {
    "american_date_instruction",
    "american_written_date",
    "american_time_12h",
    "american_customary_unit",
}
OTHER_PATTERN_KEYS = {
    "other_date_instruction",
    "other_written_date",
    "other_time_24h",
    "other_metric_unit",
}


@dataclass
class ItemRecord:
    source_bundle: str
    benchmark_family: str
    benchmark_task: str
    split: str
    item_id: str
    prompt_text: str
    gold_text: str


def normalize_space(text: str) -> str:
    return " ".join(text.replace("\u2019", "'").replace("\u2018", "'").split())


def tagged_output_path(name: str, tag: str) -> Path:
    path = OUTPUT_DIR / name
    if not tag:
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def safe_json_line_iter(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def flatten_strings(value) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        strings.append(value)
    elif isinstance(value, (int, float)):
        strings.append(str(value))
    elif isinstance(value, list):
        for item in value:
            strings.extend(flatten_strings(item))
    elif isinstance(value, dict):
        for item in value.values():
            strings.extend(flatten_strings(item))
    return strings


def iter_parquet_rows(path: Path, columns: list[str]) -> Iterable[dict]:
    connection = duckdb.connect(database=":memory:")
    quoted_path = path.as_posix().replace("'", "''")
    query = f"SELECT {', '.join(columns)} FROM read_parquet('{quoted_path}')"
    cursor = connection.execute(query)
    colnames = [desc[0] for desc in cursor.description]
    while True:
        rows = cursor.fetchmany(1000)
        if not rows:
            break
        for row in rows:
            yield dict(zip(colnames, row))
    connection.close()


def format_choices(choices: list[str]) -> str:
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    rendered = []
    for idx, choice in enumerate(choices):
        label = labels[idx] if idx < len(labels) else str(idx)
        rendered.append(f"{label}. {choice}")
    return " ".join(rendered)


def resolve_choice_from_label(choices: list[str], label) -> str:
    if label is None:
        return ""
    label_str = str(label).strip()
    if label_str.isdigit():
        idx = int(label_str)
        if 0 <= idx < len(choices):
            return choices[idx]
        if 1 <= idx <= len(choices):
            return choices[idx - 1]
    upper = label_str.upper()
    if len(upper) == 1 and "A" <= upper <= "Z":
        idx = ord(upper) - ord("A")
        if 0 <= idx < len(choices):
            return choices[idx]
    return label_str


def iter_benchmarks() -> Iterable[ItemRecord]:
    # BIG-bench: date_understanding
    path = DATASETS / "bigbench" / "date_understanding" / "task.json"
    obj = json.loads(path.read_text())
    for idx, example in enumerate(obj["examples"]):
        positive = [k for k, v in example["target_scores"].items() if v == 1]
        yield ItemRecord(
            source_bundle="local",
            benchmark_family="BIG-bench",
            benchmark_task="date_understanding",
            split="eval",
            item_id=f"date_understanding_{idx}",
            prompt_text=normalize_space(example["input"]),
            gold_text=normalize_space(" ".join(positive)),
        )

    # BIG-bench: temporal_sequences
    path = DATASETS / "bigbench" / "temporal_sequences" / "task.json"
    obj = json.loads(path.read_text())
    for idx, example in enumerate(obj["examples"]):
        positive = [example.get("target", "")] or [k for k, v in example["target_scores"].items() if v == 1]
        yield ItemRecord(
            source_bundle="local",
            benchmark_family="BIG-bench",
            benchmark_task="temporal_sequences",
            split="eval",
            item_id=f"temporal_sequences_{idx}",
            prompt_text=normalize_space(example["input"]),
            gold_text=normalize_space(" ".join(positive)),
        )

    # GSM8K: evaluation split only
    path = DATASETS / "gsm8k" / "test.jsonl"
    for idx, row in enumerate(safe_json_line_iter(path)):
        yield ItemRecord(
            source_bundle="local",
            benchmark_family="GSM8K",
            benchmark_task="grade_school_math",
            split="test",
            item_id=f"gsm8k_test_{idx}",
            prompt_text=normalize_space(row["question"]),
            gold_text=normalize_space(row["answer"]),
        )

    # SVAMP
    path = DATASETS / "svamp" / "SVAMP.json"
    rows = json.loads(path.read_text())
    for idx, row in enumerate(rows):
        prompt_parts = [row.get("Body", ""), row.get("Question", "")]
        gold_parts = [str(row.get("Answer", "")), row.get("Equation", "")]
        yield ItemRecord(
            source_bundle="local",
            benchmark_family="SVAMP",
            benchmark_task="svamp",
            split="eval",
            item_id=str(row.get("ID", f"svamp_{idx}")),
            prompt_text=normalize_space(" ".join(filter(None, prompt_parts))),
            gold_text=normalize_space(" ".join(filter(None, gold_parts))),
        )

    # TIMEBENCH
    timebench_root = DATASETS / "timebench" / "full" / "TimeBench-full-19000"
    for path in sorted(timebench_root.rglob("*")):
        if not path.is_file() or path.suffix not in {".jsonl", ".json"}:
            continue
        for idx, row in enumerate(safe_json_line_iter(path)):
            prompt_parts = []
            for key in ["question", "context", "options"]:
                if key in row:
                    prompt_parts.extend(flatten_strings(row[key]))
            gold_parts = []
            for key in ["answer", "labels", "target"]:
                if key in row:
                    gold_parts.extend(flatten_strings(row[key]))
            source_task = path.relative_to(timebench_root).parts[0]
            subtask = path.stem
            item_id = row.get("idx") or row.get("qid") or f"{subtask}_{idx}"
            yield ItemRecord(
                source_bundle="local",
                benchmark_family="TIMEBENCH",
                benchmark_task=source_task,
                split=subtask,
                item_id=str(item_id),
                prompt_text=normalize_space(" ".join(filter(None, prompt_parts))),
                gold_text=normalize_space(" ".join(filter(None, gold_parts))),
            )

    # MGSM: English split only for English-variation audit
    path = DATASETS / "mgsm" / "mgsm_en.tsv"
    with path.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            question, answer = line.split("\t", 1)
            yield ItemRecord(
                source_bundle="local",
                benchmark_family="MGSM",
                benchmark_task="mgsm_en",
                split="eval",
                item_id=f"mgsm_en_{idx}",
                prompt_text=normalize_space(question),
                gold_text=normalize_space(answer),
            )

    # Hugging Face: MMLU
    for path in sorted((DATASETS_HF / "cais__mmlu" / "all").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["question", "subject", "choices", "answer"])):
            choices = flatten_strings(row.get("choices", []))
            prompt = row["question"]
            if choices:
                prompt = f"{prompt} Choices: {format_choices(choices)}"
            answer_idx = row.get("answer")
            gold = choices[answer_idx] if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices) else str(answer_idx)
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="MMLU",
                benchmark_task="all",
                split=split,
                item_id=f"mmlu_{split}_{idx}",
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: HellaSwag
    for path in sorted((DATASETS_HF / "Rowan__hellaswag" / "data").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["ctx", "endings", "label", "ind"])):
            endings = flatten_strings(row.get("endings", []))
            prompt = row["ctx"]
            if endings:
                prompt = f"{prompt} Endings: {format_choices(endings)}"
            label = row.get("label")
            gold = endings[int(label)] if str(label).isdigit() and int(label) < len(endings) else str(label)
            item_id = row.get("ind", f"hellaswag_{split}_{idx}")
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="HellaSwag",
                benchmark_task="default",
                split=split,
                item_id=str(item_id),
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: DROP
    for path in sorted((DATASETS_HF / "ucinlp__drop" / "data").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["section_id", "query_id", "passage", "question", "answers_spans"])):
            answers_struct = row.get("answers_spans", {}) or {}
            answers = flatten_strings(answers_struct.get("spans", []))
            prompt = f"{row['passage']} {row['question']}"
            gold = " ".join(answers)
            item_id = row.get("query_id") or f"drop_{split}_{idx}"
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="DROP",
                benchmark_task="default",
                split=split,
                item_id=str(item_id),
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: BBH
    bbh_root = DATASETS_HF / "lukaemon__bbh"
    for path in sorted(bbh_root.rglob("*.parquet")):
        task = path.parent.name
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["input", "target"])):
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="BBH",
                benchmark_task=task,
                split=split,
                item_id=f"bbh_{task}_{idx}",
                prompt_text=normalize_space(row["input"]),
                gold_text=normalize_space(str(row["target"])),
            )

    # Hugging Face: ARC-Challenge
    for path in sorted((DATASETS_HF / "allenai__ai2_arc" / "ARC-Challenge").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["id", "question", "choices", "answerKey"])):
            choices = row.get("choices", {}) or {}
            choice_texts = flatten_strings(choices.get("text", []))
            choice_labels = flatten_strings(choices.get("label", []))
            prompt = row["question"]
            if choice_texts:
                rendered = [f"{label}. {text}" for label, text in zip(choice_labels, choice_texts)]
                prompt = f"{prompt} Choices: {' '.join(rendered)}"
            gold = row.get("answerKey", "")
            for label, text in zip(choice_labels, choice_texts):
                if label == row.get("answerKey"):
                    gold = text
                    break
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="ARC",
                benchmark_task="ARC-Challenge",
                split=split,
                item_id=str(row.get("id", f"arc_{split}_{idx}")),
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(str(gold)),
            )

    # Hugging Face: TruthfulQA
    truth_root = DATASETS_HF / "truthfulqa__truthful_qa"
    generation_path = truth_root / "generation" / "validation-00000-of-00001.parquet"
    for idx, row in enumerate(iter_parquet_rows(generation_path, ["question", "best_answer", "correct_answers"])):
        gold_parts = [row.get("best_answer", "")] + flatten_strings(row.get("correct_answers", []))
        yield ItemRecord(
            source_bundle="huggingface",
            benchmark_family="TruthfulQA",
            benchmark_task="generation",
            split="validation",
            item_id=f"truthfulqa_generation_{idx}",
            prompt_text=normalize_space(row["question"]),
            gold_text=normalize_space(" ".join(filter(None, gold_parts))),
        )

    mc_path = truth_root / "multiple_choice" / "validation-00000-of-00001.parquet"
    for idx, row in enumerate(iter_parquet_rows(mc_path, ["question", "mc1_targets"])):
        targets = row.get("mc1_targets", {}) or {}
        choices = flatten_strings(targets.get("choices", []))
        labels = list(targets.get("labels", [])) if isinstance(targets.get("labels", []), list) else []
        prompt = row["question"]
        if choices:
            prompt = f"{prompt} Choices: {format_choices(choices)}"
        gold_choices = [choice for choice, label in zip(choices, labels) if label == 1]
        yield ItemRecord(
            source_bundle="huggingface",
            benchmark_family="TruthfulQA",
            benchmark_task="multiple_choice",
            split="validation",
            item_id=f"truthfulqa_mc_{idx}",
            prompt_text=normalize_space(prompt),
            gold_text=normalize_space(" ".join(gold_choices)),
        )

    # Hugging Face: WinoGrande XL
    for path in sorted((DATASETS_HF / "winogrande" / "winogrande_xl").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["sentence", "option1", "option2", "answer"])):
            prompt = f"{row['sentence']} Options: A. {row['option1']} B. {row['option2']}"
            gold = row["option1"] if row.get("answer") == "1" else row["option2"] if row.get("answer") == "2" else str(row.get("answer", ""))
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="WinoGrande",
                benchmark_task="winogrande_xl",
                split=split,
                item_id=f"winogrande_{split}_{idx}",
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: MMLU-Pro
    for path in sorted((DATASETS_HF / "TIGER-Lab__MMLU-Pro" / "data").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["question_id", "question", "options", "answer", "answer_index", "category"])):
            choices = flatten_strings(row.get("options", []))
            prompt = row["question"]
            if choices:
                prompt = f"{prompt} Choices: {format_choices(choices)}"
            answer_idx = row.get("answer_index")
            gold = choices[answer_idx] if isinstance(answer_idx, int) and 0 <= answer_idx < len(choices) else resolve_choice_from_label(choices, row.get("answer", ""))
            item_id = row.get("question_id", f"mmlu_pro_{split}_{idx}")
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="MMLU-Pro",
                benchmark_task="default",
                split=split,
                item_id=str(item_id),
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: BoolQ
    for path in sorted((DATASETS_HF / "google__boolq" / "data").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["question", "answer", "passage"])):
            prompt = f"{row['passage']} Question: {row['question']}"
            gold = "True" if row.get("answer") else "False"
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="BoolQ",
                benchmark_task="default",
                split=split,
                item_id=f"boolq_{split}_{idx}",
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: TriviaQA (rc.nocontext)
    for path in sorted((DATASETS_HF / "mandarjoshi__trivia_qa" / "rc.nocontext").glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        for idx, row in enumerate(iter_parquet_rows(path, ["question", "question_id", "answer"])):
            answer_struct = row.get("answer", {}) or {}
            gold_parts = []
            for key in ["value", "normalized_value", "aliases", "normalized_aliases"]:
                if key in answer_struct:
                    gold_parts.extend(flatten_strings(answer_struct.get(key)))
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="TriviaQA",
                benchmark_task="rc_nocontext",
                split=split,
                item_id=str(row.get("question_id", f"triviaqa_{split}_{idx}")),
                prompt_text=normalize_space(row["question"]),
                gold_text=normalize_space(" ".join(filter(None, gold_parts))),
            )

    # Hugging Face: AGIEval (development splits)
    agieval_root = DATASETS_HF / "RUCAIBox__AGIEval" / "dev"
    for path in sorted(agieval_root.glob("*.jsonl")):
        task = path.stem
        for idx, row in enumerate(safe_json_line_iter(path)):
            prompt_parts = []
            for key in ["passage", "question", "context"]:
                if key in row:
                    prompt_parts.extend(flatten_strings(row.get(key)))
            options = flatten_strings(row.get("options", []))
            if options:
                prompt_parts.append("Choices: " + " ".join(options))
            gold = ""
            if "label" in row:
                gold = resolve_choice_from_label(options, row.get("label"))
            elif "answer" in row:
                gold = resolve_choice_from_label(options, row.get("answer"))
            elif "gold" in row:
                gold = resolve_choice_from_label(options, row.get("gold"))
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="AGIEval",
                benchmark_task=task,
                split="dev",
                item_id=f"agieval_{task}_{idx}",
                prompt_text=normalize_space(" ".join(filter(None, prompt_parts))),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: MedQA USMLE 4-option
    medqa_root = DATASETS_HF / "GBaker__MedQA-USMLE-4-options-hf"
    for path in [medqa_root / "dev.json", medqa_root / "test.json"]:
        split = path.stem
        for idx, row in enumerate(safe_json_line_iter(path)):
            choices = [row.get(f"ending{i}", "") for i in range(4)]
            prompt_parts = [row.get("sent1", ""), row.get("sent2", "")]
            prompt = " ".join(filter(None, prompt_parts))
            if any(choices):
                prompt = f"{prompt} Choices: {format_choices(choices)}"
            gold = choices[row["label"]] if isinstance(row.get("label"), int) and 0 <= row["label"] < len(choices) else resolve_choice_from_label(choices, row.get("label"))
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="MedQA",
                benchmark_task="usmle_4_options",
                split=split,
                item_id=str(row.get("id", f"medqa_{split}_{idx}")),
                prompt_text=normalize_space(prompt),
                gold_text=normalize_space(gold),
            )

    # Hugging Face: IFEval
    ifeval_path = DATASETS_HF / "google__IFEval" / "ifeval_input_data.jsonl"
    for idx, row in enumerate(safe_json_line_iter(ifeval_path)):
        yield ItemRecord(
            source_bundle="huggingface",
            benchmark_family="IFEval",
            benchmark_task="default",
            split="eval",
            item_id=str(row.get("key", f"ifeval_{idx}")),
            prompt_text=normalize_space(row.get("prompt", "")),
            gold_text="",
        )

    # Hugging Face: HumanEval
    humaneval_path = DATASETS_HF / "openai__openai_humaneval" / "openai_humaneval" / "test-00000-of-00001.parquet"
    for idx, row in enumerate(iter_parquet_rows(humaneval_path, ["task_id", "prompt", "canonical_solution", "test"])):
        gold = " ".join(filter(None, [row.get("canonical_solution", ""), row.get("test", "")]))
        yield ItemRecord(
            source_bundle="huggingface",
            benchmark_family="HumanEval",
            benchmark_task="default",
            split="test",
            item_id=str(row.get("task_id", f"humaneval_{idx}")),
            prompt_text=normalize_space(row.get("prompt", "")),
            gold_text=normalize_space(gold),
        )

    # Hugging Face: MBPP sanitized
    mbpp_root = DATASETS_HF / "google-research-datasets__mbpp" / "sanitized"
    for path in sorted(mbpp_root.glob("*.parquet")):
        split = path.name.split("-", 1)[0]
        if split not in {"test", "validation"}:
            continue
        for idx, row in enumerate(iter_parquet_rows(path, ["task_id", "prompt", "code", "test_list"])):
            gold_parts = [row.get("code", "")]
            gold_parts.extend(flatten_strings(row.get("test_list", [])))
            yield ItemRecord(
                source_bundle="huggingface",
                benchmark_family="MBPP",
                benchmark_task="sanitized",
                split=split,
                item_id=str(row.get("task_id", f"mbpp_{split}_{idx}")),
                prompt_text=normalize_space(row.get("prompt", "")),
                gold_text=normalize_space(" ".join(filter(None, gold_parts))),
            )


def build_term_maps() -> tuple[re.Pattern, re.Pattern, dict[str, list[dict]], dict[str, list[dict]]]:
    rows = json.loads((RESOURCES / "us_uk_variations_categorized.json").read_text())
    us_terms: dict[str, list[dict]] = defaultdict(list)
    other_terms: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        us = row["us"].lower()
        other = row["uk"].lower()
        us_terms[us].append(row)
        other_terms[other].append(row)

    def compile_pattern(terms: Iterable[str]) -> re.Pattern:
        ordered = sorted(set(terms), key=len, reverse=True)
        escaped = [re.escape(term) for term in ordered]
        return re.compile(r"(?<![A-Za-z])(?:" + "|".join(escaped) + r")(?![A-Za-z])", re.IGNORECASE)

    return compile_pattern(us_terms), compile_pattern(other_terms), us_terms, other_terms


def find_inventory_matches(
    text: str,
    us_pattern: re.Pattern,
    other_pattern: re.Pattern,
    us_terms: dict[str, list[dict]],
    other_terms: dict[str, list[dict]],
) -> tuple[list[dict], list[dict]]:
    us_hits = []
    other_hits = []
    for match in us_pattern.finditer(text):
        term = match.group(0).lower()
        for meta in us_terms.get(term, []):
            us_hits.append(
                {
                    "term": term,
                    "pair_term": meta["uk"],
                    "category": meta.get("category", "unknown"),
                    "source": "inventory",
                }
            )
    for match in other_pattern.finditer(text):
        term = match.group(0).lower()
        for meta in other_terms.get(term, []):
            other_hits.append(
                {
                    "term": term,
                    "pair_term": meta["us"],
                    "category": meta.get("category", "unknown"),
                    "source": "inventory",
                }
            )
    return us_hits, other_hits


def find_convention_matches(text: str) -> tuple[list[dict], list[dict]]:
    american_hits = []
    other_hits = []
    for label, pattern in CONVENTION_PATTERNS.items():
        for match in pattern.finditer(text):
            payload = {
                "term": match.group(0),
                "pair_term": "",
                "category": label,
                "source": "convention",
            }
            if label in AMERICAN_PATTERN_KEYS:
                american_hits.append(payload)
            elif label in OTHER_PATTERN_KEYS:
                other_hits.append(payload)
    return american_hits, other_hits


def dedupe_hits(hits: list[dict]) -> list[dict]:
    seen = set()
    unique = []
    for hit in hits:
        key = (hit["term"].lower(), hit["category"], hit["source"], hit["pair_term"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique


def classify(american_count: int, other_count: int) -> str:
    if american_count and other_count:
        return "mixed"
    if american_count:
        return "american_only"
    if other_count:
        return "other_only"
    return "no_match"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan benchmark gold data for American-centered vs other user-valid signals.")
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix for output filenames, e.g. provider_expanded -> variation_summary_provider_expanded.md",
    )
    parser.add_argument(
        "--only-families",
        default="",
        help="Optional comma-separated benchmark-family filter, e.g. MMLU-Pro,AGIEval,BoolQ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    us_pattern, other_pattern, us_terms, other_terms = build_term_maps()
    allowed_families = {part.strip() for part in args.only_families.split(",") if part.strip()}

    filtered_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []
    term_counter = Counter()
    benchmark_stats: dict[tuple[str, str], Counter] = defaultdict(Counter)

    all_items = 0
    flagged_items = 0
    overall = Counter()

    for item in iter_benchmarks():
        if allowed_families and item.benchmark_family not in allowed_families:
            continue
        all_items += 1
        full_text = normalize_space(f"{item.prompt_text} {item.gold_text}")
        american_inventory_hits, other_inventory_hits = find_inventory_matches(full_text, us_pattern, other_pattern, us_terms, other_terms)
        american_convention_hits, other_convention_hits = find_convention_matches(full_text)

        american_hits = dedupe_hits(american_inventory_hits + american_convention_hits)
        other_hits = dedupe_hits(other_inventory_hits + other_convention_hits)

        if not american_hits and not other_hits:
            benchmark_stats[(item.benchmark_family, item.benchmark_task)]["total_items"] += 1
            continue

        flagged_items += 1
        benchmark_stats[(item.benchmark_family, item.benchmark_task)]["total_items"] += 1
        benchmark_stats[(item.benchmark_family, item.benchmark_task)]["flagged_items"] += 1

        american_count = len(american_hits)
        other_count = len(other_hits)
        classification = classify(american_count, other_count)

        overall["american_match_count"] += american_count
        overall["other_match_count"] += other_count
        overall[classification] += 1

        stats = benchmark_stats[(item.benchmark_family, item.benchmark_task)]
        stats["american_match_count"] += american_count
        stats["other_match_count"] += other_count
        stats[classification] += 1

        for hit in american_hits:
            term_counter[(item.benchmark_family, item.benchmark_task, "american", hit["category"], hit["term"].lower())] += 1
        for hit in other_hits:
            term_counter[(item.benchmark_family, item.benchmark_task, "other", hit["category"], hit["term"].lower())] += 1

        filtered_rows.append(
            {
                "source_bundle": item.source_bundle,
                "benchmark_family": item.benchmark_family,
                "benchmark_task": item.benchmark_task,
                "split": item.split,
                "item_id": item.item_id,
                "classification": classification,
                "american_match_count": str(american_count),
                "other_match_count": str(other_count),
                "american_matches": json.dumps(american_hits, ensure_ascii=True),
                "other_matches": json.dumps(other_hits, ensure_ascii=True),
                "prompt_text": item.prompt_text,
                "gold_text": item.gold_text,
            }
        )

    for (family, task), stats in sorted(benchmark_stats.items()):
        total_items = stats["total_items"]
        flagged = stats["flagged_items"]
        american_total = stats["american_match_count"]
        other_total = stats["other_match_count"]
        total_matches = american_total + other_total
        summary_rows.append(
            {
                "benchmark_family": family,
                "benchmark_task": task,
                "total_items": str(total_items),
                "flagged_items": str(flagged),
                "flagged_pct": f"{(100 * flagged / total_items):.2f}" if total_items else "0.00",
                "american_only_items": str(stats["american_only"]),
                "other_only_items": str(stats["other_only"]),
                "mixed_items": str(stats["mixed"]),
                "american_match_count": str(american_total),
                "other_match_count": str(other_total),
                "american_match_pct": f"{(100 * american_total / total_matches):.2f}" if total_matches else "0.00",
                "other_match_pct": f"{(100 * other_total / total_matches):.2f}" if total_matches else "0.00",
            }
        )

    term_rows = []
    for (family, task, side, category, term), count in term_counter.most_common():
        term_rows.append(
            {
                "benchmark_family": family,
                "benchmark_task": task,
                "side": side,
                "category": category,
                "term": term,
                "count": str(count),
            }
        )

    with tagged_output_path("filtered_variation_samples.csv", args.tag).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(filtered_rows[0]) if filtered_rows else [
            "source_bundle", "benchmark_family", "benchmark_task", "split", "item_id", "classification",
            "american_match_count", "other_match_count", "american_matches", "other_matches",
            "prompt_text", "gold_text"
        ])
        writer.writeheader()
        writer.writerows(filtered_rows)

    with tagged_output_path("variation_summary_by_benchmark.csv", args.tag).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]) if summary_rows else [
            "benchmark_family", "benchmark_task", "total_items", "flagged_items", "flagged_pct",
            "american_only_items", "other_only_items", "mixed_items",
            "american_match_count", "other_match_count", "american_match_pct", "other_match_pct"
        ])
        writer.writeheader()
        writer.writerows(summary_rows)

    with tagged_output_path("variation_term_frequencies.csv", args.tag).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(term_rows[0]) if term_rows else [
            "benchmark_family", "benchmark_task", "side", "category", "term", "count"
        ])
        writer.writeheader()
        writer.writerows(term_rows)

    total_matches = overall["american_match_count"] + overall["other_match_count"]
    lines = [
        "# Benchmark Variation Scan Summary",
        "",
        f"- Total benchmark items scanned: {all_items}",
        f"- Items with at least one American-or-other signal: {flagged_items} ({(100 * flagged_items / all_items):.2f}%)" if all_items else "- Items with at least one American-or-other signal: 0",
        f"- American-only flagged items: {overall['american_only']}",
        f"- Other-only flagged items: {overall['other_only']}",
        f"- Mixed flagged items: {overall['mixed']}",
        f"- American-side match share: {(100 * overall['american_match_count'] / total_matches):.2f}%" if total_matches else "- American-side match share: 0.00%",
        f"- Other-side match share: {(100 * overall['other_match_count'] / total_matches):.2f}%" if total_matches else "- Other-side match share: 0.00%",
        "",
        "## By benchmark",
        "",
        "| Benchmark | Task | Items | Flagged | American only | Other only | Mixed | American match % | Other match % |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['benchmark_family']} | {row['benchmark_task']} | {row['total_items']} | {row['flagged_items']} ({row['flagged_pct']}%) | {row['american_only_items']} | {row['other_only_items']} | {row['mixed_items']} | {row['american_match_pct']} | {row['other_match_pct']} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `american_match_pct` and `other_match_pct` are computed over detected signals, not over all items.",
            "- `other` is a broad umbrella label in the output, but the seed lexical inventory is derived from AmE-vs-UK pairs. The convention detector broadens coverage to non-American date, time, and measurement patterns.",
            "- For this first pass, only the English split of `MGSM` is scanned for English-variation signals.",
            "- The Hugging Face bundle is scanned over the locally downloaded parquet files in `datasets_hf/`, including train/validation/test splits where available because this is a gold-data audit rather than a model-score reproduction run.",
        ]
    )
    (tagged_output_path("variation_summary.md", args.tag)).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
