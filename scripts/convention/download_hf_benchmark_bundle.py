from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb
from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets_hf"
BENCHMARK_SOURCES_DIR = ROOT / "data" / "benchmark_sources"


DOWNLOAD_SPECS = [
    {
        "benchmark_family": "MMLU",
        "benchmark_task": "all",
        "repo_id": "cais/mmlu",
        "coverage": "Massively reused knowledge benchmark with natural-language multiple-choice questions; high-value for lexical and convention auditing.",
        "files": [
            "README.md",
            "all/dev-00000-of-00001.parquet",
            "all/test-00000-of-00001.parquet",
            "all/validation-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "HellaSwag",
        "benchmark_task": "default",
        "repo_id": "Rowan/hellaswag",
        "coverage": "Widely used commonsense benchmark with rich natural-language endings; useful for wording and narrative-style variation.",
        "files": [
            "README.md",
            "data/train-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "DROP",
        "benchmark_task": "default",
        "repo_id": "ucinlp/drop",
        "coverage": "Reading-comprehension benchmark rich in dates, numbers, and temporal expressions; strong fit for date/time normalization audits.",
        "files": [
            "README.md",
            "data/train-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "BBH",
        "benchmark_task": "all_tasks",
        "repo_id": "lukaemon/bbh",
        "coverage": "High-visibility instruction-style reasoning benchmark with many subtask formats; useful expansion beyond the two BIG-bench tasks already downloaded.",
        "files": [
            "README.md",
            "boolean_expressions/test-00000-of-00001.parquet",
            "causal_judgement/test-00000-of-00001.parquet",
            "date_understanding/test-00000-of-00001.parquet",
            "disambiguation_qa/test-00000-of-00001.parquet",
            "dyck_languages/test-00000-of-00001.parquet",
            "formal_fallacies/test-00000-of-00001.parquet",
            "geometric_shapes/test-00000-of-00001.parquet",
            "hyperbaton/test-00000-of-00001.parquet",
            "logical_deduction_five_objects/test-00000-of-00001.parquet",
            "logical_deduction_seven_objects/test-00000-of-00001.parquet",
            "logical_deduction_three_objects/test-00000-of-00001.parquet",
            "movie_recommendation/test-00000-of-00001.parquet",
            "multistep_arithmetic_two/test-00000-of-00001.parquet",
            "navigate/test-00000-of-00001.parquet",
            "object_counting/test-00000-of-00001.parquet",
            "penguins_in_a_table/test-00000-of-00001.parquet",
            "reasoning_about_colored_objects/test-00000-of-00001.parquet",
            "ruin_names/test-00000-of-00001.parquet",
            "salient_translation_error_detection/test-00000-of-00001.parquet",
            "snarks/test-00000-of-00001.parquet",
            "sports_understanding/test-00000-of-00001.parquet",
            "temporal_sequences/test-00000-of-00001.parquet",
            "tracking_shuffled_objects_five_objects/test-00000-of-00001.parquet",
            "tracking_shuffled_objects_seven_objects/test-00000-of-00001.parquet",
            "tracking_shuffled_objects_three_objects/test-00000-of-00001.parquet",
            "web_of_lies/test-00000-of-00001.parquet",
            "word_sorting/test-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "ARC",
        "benchmark_task": "ARC-Challenge",
        "repo_id": "allenai/ai2_arc",
        "coverage": "Provider-reused science QA benchmark that can expose measurement and wording defaults in short QA format.",
        "files": [
            "README.md",
            "ARC-Challenge/train-00000-of-00001.parquet",
            "ARC-Challenge/validation-00000-of-00001.parquet",
            "ARC-Challenge/test-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "TruthfulQA",
        "benchmark_task": "generation_and_mc",
        "repo_id": "truthfulqa/truthful_qa",
        "coverage": "Widely cited short-answer QA benchmark; compact and useful as a natural-language audit extension.",
        "files": [
            "README.md",
            "generation/validation-00000-of-00001.parquet",
            "multiple_choice/validation-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "WinoGrande",
        "benchmark_task": "winogrande_xl",
        "repo_id": "winogrande",
        "coverage": "Widely reused language benchmark; lower direct fit for date/time/unit analysis but useful for broader wording audit coverage.",
        "files": [
            "README.md",
            "winogrande_xl/train-00000-of-00001.parquet",
            "winogrande_xl/validation-00000-of-00001.parquet",
            "winogrande_xl/test-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "MMLU-Pro",
        "benchmark_task": "default",
        "repo_id": "TIGER-Lab/MMLU-Pro",
        "coverage": "High-value provider benchmark and stronger successor to MMLU; useful for updated audit coverage on benchmark gold text.",
        "files": [
            "README.md",
            "data/validation-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "AGIEval",
        "benchmark_task": "dev_all",
        "repo_id": "RUCAIBox/AGIEval",
        "coverage": "Provider-reported exam-style benchmark with rich question text and answer formatting; promising for auditing hidden U.S.-centric defaults.",
        "files": [
            "README.md",
            "dev/aqua-rat.jsonl",
            "dev/gaokao-biology.jsonl",
            "dev/gaokao-chemistry.jsonl",
            "dev/gaokao-chinese.jsonl",
            "dev/gaokao-english.jsonl",
            "dev/gaokao-geography.jsonl",
            "dev/gaokao-history.jsonl",
            "dev/gaokao-mathcloze.jsonl",
            "dev/gaokao-mathqa.jsonl",
            "dev/gaokao-physics.jsonl",
            "dev/jec-qa-ca.jsonl",
            "dev/jec-qa-kd.jsonl",
            "dev/lsat-ar.jsonl",
            "dev/lsat-lr.jsonl",
            "dev/lsat-rc.jsonl",
            "dev/logiqa-en.jsonl",
            "dev/logiqa-zh.jsonl",
            "dev/sat-en-without-passage.jsonl",
            "dev/sat-en.jsonl",
            "dev/sat-math.jsonl",
        ],
    },
    {
        "benchmark_family": "BoolQ",
        "benchmark_task": "default",
        "repo_id": "google/boolq",
        "coverage": "Common provider-reported QA benchmark with natural-language passages and binary gold answers; good audit extension.",
        "files": [
            "README.md",
            "data/train-00000-of-00001.parquet",
            "data/validation-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "TriviaQA",
        "benchmark_task": "rc_nocontext",
        "repo_id": "mandarjoshi/trivia_qa",
        "coverage": "Well-known QA benchmark with natural-language questions and gold answers; useful for broadening provider-aligned audit coverage without the large passage payload.",
        "files": [
            "README.md",
            "rc.nocontext/validation-00000-of-00001.parquet",
            "rc.nocontext/test-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "MedQA",
        "benchmark_task": "usmle_4_options",
        "repo_id": "GBaker/MedQA-USMLE-4-options-hf",
        "coverage": "Frequently reported medical benchmark with standardized multiple-choice gold data; useful for checking U.S.-specific language and professional-domain defaults.",
        "files": [
            "README.md",
            "dev.json",
            "test.json",
        ],
    },
    {
        "benchmark_family": "GPQA",
        "benchmark_task": "main_and_diamond",
        "repo_id": "Idavidrein/gpqa",
        "coverage": "Frontier reasoning benchmark heavily used in current provider reporting; lower direct fit for locale signals but important for provider-coverage completeness.",
        "files": [
            "README.md",
            "gpqa_main.csv",
            "gpqa_diamond.csv",
        ],
    },
    {
        "benchmark_family": "IFEval",
        "benchmark_task": "default",
        "repo_id": "google/IFEval",
        "coverage": "Instruction-following benchmark used in newer provider reporting; helpful for coverage breadth even if less central to the American-default argument.",
        "files": [
            "README.md",
            "ifeval_input_data.jsonl",
        ],
    },
    {
        "benchmark_family": "HumanEval",
        "benchmark_task": "default",
        "repo_id": "openai/openai_humaneval",
        "coverage": "Canonical coding benchmark reused across provider reports; included for provider-aligned benchmark coverage.",
        "files": [
            "README.md",
            "openai_humaneval/test-00000-of-00001.parquet",
        ],
    },
    {
        "benchmark_family": "MBPP",
        "benchmark_task": "sanitized",
        "repo_id": "google-research-datasets/mbpp",
        "coverage": "Common coding benchmark paired with HumanEval in provider reporting; useful mainly for provider-coverage completeness.",
        "files": [
            "README.md",
            "sanitized/test-00000-of-00001.parquet",
            "sanitized/validation-00000-of-00001.parquet",
        ],
    },
]


def download_file(repo_id: str, filename: str) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=DATASETS_DIR / repo_id.replace("/", "__"),
            local_dir_use_symlinks=False,
        )
    )


def parquet_row_count(path: Path) -> int:
    return duckdb.sql(f"SELECT COUNT(*) AS n FROM read_parquet('{path.as_posix()}')").fetchone()[0]


def csv_row_count(path: Path) -> int:
    quoted_path = path.as_posix().replace("'", "''")
    return duckdb.sql(f"SELECT COUNT(*) AS n FROM read_csv_auto('{quoted_path}', header=true)").fetchone()[0]


def json_row_count(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return sum(1 for line in text.splitlines() if line.strip())
    if isinstance(data, list):
        return len(data)
    if isinstance(data, dict):
        for key in ("data", "examples", "items"):
            value = data.get(key)
            if isinstance(value, list):
                return len(value)
    return 1


def jsonl_row_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def row_count(path: Path) -> int:
    if path.suffix == ".parquet":
        return parquet_row_count(path)
    if path.suffix == ".csv":
        return csv_row_count(path)
    if path.suffix == ".json":
        return json_row_count(path)
    if path.suffix == ".jsonl":
        return jsonl_row_count(path)
    return 0


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark_family",
                "benchmark_task",
                "status",
                "repo_id",
                "file_count",
                "parquet_files",
                "item_count",
                "coverage",
                "local_path",
                "source_url",
                "download_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], output_path: Path) -> None:
    lines = [
        "# Downloaded Hugging Face Benchmark Coverage",
        "",
        "This file summarizes the additional benchmark bundle downloaded from Hugging Face.",
        "",
        "| Benchmark | Task | Status | Items | Parquet files | Repo | Local path |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['benchmark_family']} | {row['benchmark_task']} | {row['status']} | {row['item_count']} | {row['parquet_files']} | `{row['repo_id']}` | `{row['local_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This bundle prioritizes benchmarks that are both widely reused in frontier-model reports and useful for the current American-default benchmark audit.",
            "- `MMLU-Pro`, `AGIEval`, `BoolQ`, `TriviaQA`, and `MedQA` are useful additional gold-data audit targets.",
            "- `GPQA`, `IFEval`, `HumanEval`, and `MBPP` are included mainly to improve coverage of benchmarks that appear in frontier-provider evaluations.",
            "- Some provider-relevant datasets may be gated on Hugging Face; when that happens the source list records the failure instead of aborting the refresh.",
            "- The files are downloaded from Hugging Face dataset repos and stored locally for reproducible auditing.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    api = HfApi()
    rows = []
    for spec in DOWNLOAD_SPECS:
        repo_dir = DATASETS_DIR / spec["repo_id"].replace("/", "__")
        repo_dir.mkdir(parents=True, exist_ok=True)

        downloaded_paths = []
        parquet_paths = []
        total_rows = 0
        status = "downloaded"
        download_error = ""

        info = api.dataset_info(spec["repo_id"])
        source_url = f"https://huggingface.co/datasets/{spec['repo_id']}/tree/{info.sha}"

        try:
            for filename in spec["files"]:
                path = download_file(spec["repo_id"], filename)
                downloaded_paths.append(path)
                if path.suffix == ".parquet":
                    parquet_paths.append(path)
                total_rows += row_count(path)
        except Exception as exc:
            status = "gated_or_failed"
            download_error = str(exc).splitlines()[0]

        rows.append(
            {
                "benchmark_family": spec["benchmark_family"],
                "benchmark_task": spec["benchmark_task"],
                "status": status,
                "repo_id": spec["repo_id"],
                "file_count": str(len(downloaded_paths)),
                "parquet_files": str(len(parquet_paths)),
                "item_count": str(total_rows),
                "coverage": spec["coverage"],
                "local_path": str(repo_dir),
                "source_url": source_url,
                "download_error": download_error,
            }
        )

    write_csv(rows, BENCHMARK_SOURCES_DIR / "downloaded_hf_benchmark_coverage.csv")
    write_markdown(rows, BENCHMARK_SOURCES_DIR / "downloaded_hf_benchmark_coverage.md")
    print(f"Downloaded {len(rows)} Hugging Face benchmark bundles into {DATASETS_DIR}")


if __name__ == "__main__":
    main()
