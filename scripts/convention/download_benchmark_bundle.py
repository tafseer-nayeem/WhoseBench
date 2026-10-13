from __future__ import annotations

import csv
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
BENCHMARK_SOURCES_DIR = ROOT / "data" / "benchmark_sources"


DOWNLOAD_SPECS = [
    {
        "benchmark_family": "BIG-bench",
        "benchmark_task": "date_understanding",
        "coverage": "Date reasoning with explicit output-format constraints; strong for auditing MM/DD/YYYY rigidity.",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/date_understanding/task.json",
                "relative_path": "bigbench/date_understanding/task.json",
            },
            {
                "url": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/date_understanding/README.md",
                "relative_path": "bigbench/date_understanding/README.md",
            },
        ],
    },
    {
        "benchmark_family": "BIG-bench",
        "benchmark_task": "temporal_sequences",
        "coverage": "Time-interval reasoning with am/pm answers; strong for auditing 12-hour defaults.",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/temporal_sequences/task.json",
                "relative_path": "bigbench/temporal_sequences/task.json",
            },
            {
                "url": "https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/temporal_sequences/README.md",
                "relative_path": "bigbench/temporal_sequences/README.md",
            },
        ],
    },
    {
        "benchmark_family": "GSM8K",
        "benchmark_task": "grade_school_math",
        "coverage": "Natural-language math word problems with rich opportunities to audit units, currency, and everyday wording defaults.",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/README.md",
                "relative_path": "gsm8k/README.md",
            },
            {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/LICENSE",
                "relative_path": "gsm8k/LICENSE",
            },
            {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl",
                "relative_path": "gsm8k/train.jsonl",
            },
            {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
                "relative_path": "gsm8k/test.jsonl",
            },
            {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train_socratic.jsonl",
                "relative_path": "gsm8k/train_socratic.jsonl",
            },
            {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test_socratic.jsonl",
                "relative_path": "gsm8k/test_socratic.jsonl",
            },
        ],
    },
    {
        "benchmark_family": "SVAMP",
        "benchmark_task": "svamp",
        "coverage": "Challenge set for math word problems built from carefully chosen variations; useful for wording and unit-sensitivity audits.",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/README.md",
                "relative_path": "svamp/README.md",
            },
            {
                "url": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/LICENSE",
                "relative_path": "svamp/LICENSE",
            },
            {
                "url": "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json",
                "relative_path": "svamp/SVAMP.json",
            },
        ],
    },
    {
        "benchmark_family": "TIMEBENCH",
        "benchmark_task": "timebench_full",
        "coverage": "Comprehensive temporal benchmark spanning symbolic, commonsense, and event temporal reasoning; best fit for date/time and time-unit audits.",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/zchuz/TimeBench/main/README.md",
                "relative_path": "timebench/README.md",
            },
            {
                "url": "https://raw.githubusercontent.com/zchuz/TimeBench/main/LICENSE",
                "relative_path": "timebench/LICENSE",
            },
            {
                "url": "https://github.com/zchuz/TimeBench/raw/main/TimeBench-full-19000.zip",
                "relative_path": "timebench/TimeBench-full-19000.zip",
                "extract_to": "timebench/full",
            },
        ],
    },
    {
        "benchmark_family": "MGSM",
        "benchmark_task": "mgsm",
        "coverage": "Multilingual grade-school math benchmark; useful as a cross-lingual extension for auditing portability of units and English-source defaults.",
        "files": [
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/README.md",
                "relative_path": "mgsm/README.md",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/LICENSE",
                "relative_path": "mgsm/LICENSE",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_bn.tsv",
                "relative_path": "mgsm/mgsm_bn.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_de.tsv",
                "relative_path": "mgsm/mgsm_de.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_en.tsv",
                "relative_path": "mgsm/mgsm_en.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_es.tsv",
                "relative_path": "mgsm/mgsm_es.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_fr.tsv",
                "relative_path": "mgsm/mgsm_fr.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_ja.tsv",
                "relative_path": "mgsm/mgsm_ja.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_ru.tsv",
                "relative_path": "mgsm/mgsm_ru.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_sw.tsv",
                "relative_path": "mgsm/mgsm_sw.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_te.tsv",
                "relative_path": "mgsm/mgsm_te.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_th.tsv",
                "relative_path": "mgsm/mgsm_th.tsv",
            },
            {
                "url": "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm/mgsm_zh.tsv",
                "relative_path": "mgsm/mgsm_zh.tsv",
            },
        ],
    },
]


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def inspect_bigbench(task_json: Path) -> tuple[int, str]:
    data = json.loads(task_json.read_text())
    examples = data.get("examples", [])
    return len(examples), "json task"


def inspect_gsm8k(folder: Path) -> tuple[int, str]:
    train = count_lines(folder / "train.jsonl")
    test = count_lines(folder / "test.jsonl")
    return train + test, f"train={train}, test={test}"


def inspect_svamp(path: Path) -> tuple[int, str]:
    data = json.loads(path.read_text())
    return len(data), "json array"


def inspect_mgsm(folder: Path) -> tuple[int, str]:
    tsv_files = sorted(folder.glob("mgsm_*.tsv"))
    total = 0
    languages = []
    for file_path in tsv_files:
        rows = count_lines(file_path)
        total += rows
        languages.append(f"{file_path.stem.split('_', 1)[1]}={rows}")
    return total, ", ".join(languages)


def inspect_timebench(folder: Path) -> tuple[int, str]:
    data_files = sorted(folder.rglob("*.jsonl")) + sorted(folder.rglob("*.json"))
    total = 0
    per_file = []
    for file_path in data_files:
        rows = count_lines(file_path)
        total += rows
        per_file.append(f"{file_path.stem}={rows}")
    return total, f"subtasks={len(data_files)}; " + ", ".join(per_file[:8]) + (" ..." if len(per_file) > 8 else "")


def maybe_extract_zip(zip_path: Path, extract_to: Path) -> None:
    if extract_to.exists() and any(extract_to.iterdir()):
        return
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(extract_to)


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in DOWNLOAD_SPECS:
        for file_spec in spec["files"]:
            destination = DATASETS_DIR / file_spec["relative_path"]
            download_file(file_spec["url"], destination)
            if "extract_to" in file_spec:
                maybe_extract_zip(destination, DATASETS_DIR / file_spec["extract_to"])

        benchmark_family = spec["benchmark_family"]
        benchmark_task = spec["benchmark_task"]

        if benchmark_family == "BIG-bench":
            item_count, detail = inspect_bigbench(DATASETS_DIR / spec["files"][0]["relative_path"])
            local_path = DATASETS_DIR / "bigbench" / benchmark_task
            data_format = "task.json"
        elif benchmark_family == "GSM8K":
            item_count, detail = inspect_gsm8k(DATASETS_DIR / "gsm8k")
            local_path = DATASETS_DIR / "gsm8k"
            data_format = "jsonl"
        elif benchmark_family == "SVAMP":
            item_count, detail = inspect_svamp(DATASETS_DIR / "svamp" / "SVAMP.json")
            local_path = DATASETS_DIR / "svamp"
            data_format = "json"
        elif benchmark_family == "TIMEBENCH":
            item_count, detail = inspect_timebench(DATASETS_DIR / "timebench" / "full")
            local_path = DATASETS_DIR / "timebench" / "full"
            data_format = "jsonl in zip"
        elif benchmark_family == "MGSM":
            item_count, detail = inspect_mgsm(DATASETS_DIR / "mgsm")
            local_path = DATASETS_DIR / "mgsm"
            data_format = "tsv"
        else:
            raise ValueError(f"Unhandled benchmark family: {benchmark_family}")

        rows.append(
            {
                "benchmark_family": benchmark_family,
                "benchmark_task": benchmark_task,
                "item_count": str(item_count),
                "coverage": spec["coverage"],
                "data_format": data_format,
                "inspection_detail": detail,
                "local_path": str(local_path),
                "official_sources": " | ".join(file_spec["url"] for file_spec in spec["files"]),
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark_family",
        "benchmark_task",
        "item_count",
        "coverage",
        "data_format",
        "inspection_detail",
        "local_path",
        "official_sources",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Downloaded Benchmark Coverage",
        "",
        "This file is generated from the locally downloaded benchmark bundle in `datasets/`.",
        "",
        "| Benchmark | Task | Items | Coverage | Format | Local path |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark_family']} | {row['benchmark_task']} | {row['item_count']} | {row['coverage']} | {row['data_format']} | `{row['local_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `date_understanding` is especially useful for date-format rigidity because the task instruction fixes answers to `MM/DD/YYYY`.",
            "- `temporal_sequences` is especially useful for time-format auditing because the answers are written as `am/pm` ranges.",
            "- `GSM8K` and `SVAMP` are the best natural-language sources for unit, currency, and wording audits.",
            "- `TIMEBENCH` is the broadest temporal suite in this bundle and is the best anchor for time/date analysis.",
            "- `MGSM` is included as an optional cross-lingual extension to test whether English-source defaults carry across languages.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = build_rows()
    write_csv(rows, BENCHMARK_SOURCES_DIR / "downloaded_benchmark_coverage.csv")
    write_markdown(rows, BENCHMARK_SOURCES_DIR / "downloaded_benchmark_coverage.md")
    print(f"Downloaded {len(rows)} benchmark bundles into {DATASETS_DIR}")


if __name__ == "__main__":
    main()
