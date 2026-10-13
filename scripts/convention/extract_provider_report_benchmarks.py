from __future__ import annotations

import csv
import re
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORT_DIR = DATA_DIR / "provider_reports"
BENCHMARK_SOURCES_DIR = ROOT / "data" / "benchmark_sources"


REPORTS = [
    {
        "provider": "OpenAI",
        "report_title": "GPT-4 Technical Report",
        "report_date": "2023-03-27",
        "source_url": "https://cdn.openai.com/papers/gpt-4.pdf",
        "filename": "openai_gpt4.pdf",
    },
    {
        "provider": "Anthropic",
        "report_title": "The Claude 3 Model Family: Opus, Sonnet, Haiku",
        "report_date": "2024-03-04",
        "source_url": "https://assets.anthropic.com/m/61e7d27f8c8f5919/original/Claude-3-Model-Card.pdf",
        "filename": "anthropic_claude3.pdf",
    },
    {
        "provider": "Google DeepMind",
        "report_title": "Gemini: A Family of Highly Capable Multimodal Models",
        "report_date": "2023-12-06",
        "source_url": "https://deepmind.google/gemini/gemini_1_report.pdf",
        "filename": "google_gemini1.pdf",
    },
    {
        "provider": "Meta",
        "report_title": "The Llama 3 Herd of Models",
        "report_date": "2024-07-23",
        "source_url": "https://arxiv.org/pdf/2407.21783.pdf",
        "filename": "meta_llama3.pdf",
    },
]


BENCHMARK_PATTERNS = {
    "MMLU": re.compile(r"\bMMLU\b"),
    "MMLU-Pro": re.compile(r"\bMMLU[- ]Pro\b", re.IGNORECASE),
    "GSM8K": re.compile(r"\bGSM[- ]?8K\b", re.IGNORECASE),
    "MGSM": re.compile(r"\bMGSM\b"),
    "HumanEval": re.compile(r"\bHumanEval\b", re.IGNORECASE),
    "MBPP": re.compile(r"\bMBPP\b"),
    "HellaSwag": re.compile(r"\bHellaSwag\b", re.IGNORECASE),
    "ARC-Challenge": re.compile(r"\bARC[- ]Challenge\b", re.IGNORECASE),
    "GPQA": re.compile(r"\bGPQA\b"),
    "BBH": re.compile(r"\bBIG[- ]Bench Hard\b|\bBBH\b", re.IGNORECASE),
    "IFEval": re.compile(r"\bIFEval\b", re.IGNORECASE),
    "TruthfulQA": re.compile(r"\bTruthfulQA\b", re.IGNORECASE),
    "TriviaQA": re.compile(r"\bTriviaQA\b", re.IGNORECASE),
    "DROP": re.compile(r"\bDROP\b"),
    "WinoGrande": re.compile(r"\bWinoGrande\b", re.IGNORECASE),
    "PIQA": re.compile(r"\bPIQA\b"),
    "SIQA": re.compile(r"\bSIQA\b"),
    "BoolQ": re.compile(r"\bBoolQ\b", re.IGNORECASE),
    "MATH": re.compile(r"\bMATH\b"),
    "AGIEval": re.compile(r"\bAGIEval\b", re.IGNORECASE),
    "MMMU": re.compile(r"\bMMMU\b"),
    "MathVista": re.compile(r"\bMathVista\b", re.IGNORECASE),
    "Natural2Code": re.compile(r"\bNatural2Code\b", re.IGNORECASE),
    "ZeroSCROLLS": re.compile(r"\bZeroSCROLLS\b", re.IGNORECASE),
    "QuALITY": re.compile(r"\bQuALITY\b"),
    "InfiniteBench": re.compile(r"\bInfiniteBench\b", re.IGNORECASE),
    "BFCL": re.compile(r"\bBFCL\b"),
    "Nexus": re.compile(r"\bNexus\b"),
    "MUSR": re.compile(r"\bMUSR\b"),
    "Math-StackExchange": re.compile(r"\bMath[- ]StackExchange\b", re.IGNORECASE),
}


def ensure_download(report: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / report["filename"]
    if not path.exists():
        urllib.request.urlretrieve(report["source_url"], path)
    return path


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks = []
    for page in reader.pages:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def main() -> None:
    rows: list[dict[str, str]] = []
    per_benchmark_providers: dict[str, set[str]] = defaultdict(set)

    for report in REPORTS:
        path = ensure_download(report)
        text = extract_pdf_text(path)
        text_path = path.with_suffix(".txt")
        text_path.write_text(text, encoding="utf-8")

        found = []
        for benchmark, pattern in BENCHMARK_PATTERNS.items():
            matches = pattern.findall(text)
            if matches:
                found.append((benchmark, len(matches)))
                per_benchmark_providers[benchmark].add(report["provider"])

        found.sort(key=lambda item: (-item[1], item[0]))
        rows.append(
            {
                "provider": report["provider"],
                "report_title": report["report_title"],
                "report_date": report["report_date"],
                "benchmarks_found": "; ".join(name for name, _ in found),
                "benchmark_count": str(len(found)),
                "pdf_path": str(path),
                "text_path": str(text_path),
                "source_url": report["source_url"],
            }
        )

    BENCHMARK_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    with (BENCHMARK_SOURCES_DIR / "provider_report_benchmarks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "provider",
                "report_title",
                "report_date",
                "benchmark_count",
                "benchmarks_found",
                "pdf_path",
                "text_path",
                "source_url",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    frequency_rows = []
    for benchmark, providers in sorted(per_benchmark_providers.items(), key=lambda item: (-len(item[1]), item[0])):
        frequency_rows.append(
            {
                "benchmark": benchmark,
                "provider_count": str(len(providers)),
                "providers": "; ".join(sorted(providers)),
            }
        )

    with (BENCHMARK_SOURCES_DIR / "provider_benchmark_frequency.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["benchmark", "provider_count", "providers"])
        writer.writeheader()
        writer.writerows(frequency_rows)

    lines = [
        "# Provider Report Benchmark File list",
        "",
        "Benchmarks were extracted from official provider reports using PDF text extraction and regex matching over a curated benchmark-name list.",
        "",
        "## Reports",
        "",
        "| Provider | Report | Date | Benchmarks found |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(f"| {row['provider']} | {row['report_title']} | {row['report_date']} | {row['benchmarks_found']} |")
    lines.extend(
        [
            "",
            "## Most reused benchmarks across providers",
            "",
            "| Benchmark | Providers |",
            "| --- | --- |",
        ]
    )
    for row in frequency_rows:
        lines.append(f"| {row['benchmark']} ({row['provider_count']}) | {row['providers']} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This source list is based on benchmark-name matching in official reports, so it is best treated as a conservative starting point rather than a perfect census.",
            "- `MATH` and a few other all-caps benchmark names may still be slightly noisy, but they are useful for identifying the most commonly reused evaluation sets.",
        ]
    )
    (BENCHMARK_SOURCES_DIR / "provider_report_benchmarks.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
