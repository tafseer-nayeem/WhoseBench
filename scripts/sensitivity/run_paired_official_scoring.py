from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset

from build_scorer_control_sensitivity import control_variants
from run_exact_match_sensitivity import exact_match
from third_party.drop_official import answer_json_to_strings, get_metrics


ROOT = Path(__file__).resolve().parents[2]
DROP_URL = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/drop/drop_dataset.zip"
DROP_SHA256 = "39d2278a29fd729de301b111a45f434c24834f40df8f4ff116d864589e3249d6"
TRUTHFULQA_REVISION = "741b8276f2d1982aa3d5b832d3ee81ed3b896490"
TRUTHFULQA_EXCLUSIONS = {
    "truthfulqa_15": "Official organization name is World Health Organization",
    "truthfulqa_762": "Flavor-Aid is a brand name, not a regional spelling variant",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No records to write: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def load_drop_questions(archive_path: Path | None) -> dict[str, dict]:
    data = archive_path.read_bytes() if archive_path else urllib.request.urlopen(DROP_URL, timeout=60).read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != DROP_SHA256:
        raise ValueError(f"DROP archive checksum mismatch: {digest}")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        raw = json.loads(archive.read("drop_dataset/drop_dataset_dev.json"))
    return {question["query_id"]: question for passage in raw.values() for question in passage["qa_pairs"]}


def drop_scores(text: str, gold_answers: list[tuple[str, ...]]) -> tuple[float, float]:
    scores = [get_metrics(text, gold) for gold in gold_answers]
    return max(score[0] for score in scores), max(score[1] for score in scores)


def bootstrap_delta(rows: list[dict[str, object]], field: str, samples: int, seed: str) -> tuple[float, float]:
    by_item: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_item[str(row["item_id"])].append(row)
    keys = sorted(by_item)
    rng = random.Random(seed)
    deltas = []
    for _ in range(samples):
        sampled = [record for key in (rng.choice(keys) for _ in keys) for record in by_item[key]]
        deltas.append(float(np.mean([float(record[f"variant_{field}"]) - float(record[f"original_{field}"]) for record in sampled])))
    return tuple(float(value) for value in np.quantile(deltas, [0.025, 0.975]))


def summarize(rows: list[dict[str, object]], benchmark: str, metric: str, group: str, samples: int, seed: int) -> dict[str, object]:
    original = float(np.mean([float(row[f"original_{metric}"]) for row in rows]))
    variant = float(np.mean([float(row[f"variant_{metric}"]) for row in rows]))
    low, high = bootstrap_delta(rows, metric, samples, f"{seed}:{benchmark}:{metric}:{group}")
    return {
        "benchmark": benchmark,
        "metric": metric,
        "group": group,
        "n_pairs": len(rows),
        "n_source_items": len({str(row["item_id"]) for row in rows}),
        "original_score": original,
        "variant_score": variant,
        "delta": variant - original,
        "ci_low": low,
        "ci_high": high,
    }


def score_drop(records: list[dict[str, str]], questions: dict[str, dict]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    scored = []
    controls = []
    for row in records:
        if row["benchmark"] != "DROP":
            continue
        question = questions[row["item_id"]]
        annotations = [question["answer"], *question.get("validated_answers", [])]
        gold_answers = [answer_json_to_strings(annotation)[0] for annotation in annotations]
        original_em, original_f1 = drop_scores(row["original_answer"], gold_answers)
        variant_em, variant_f1 = drop_scores(row["variant_answer"], gold_answers)
        eligible = original_em == 1.0 and original_f1 == 1.0
        scored.append({
            "benchmark": "DROP",
            "item_id": row["item_id"],
            "convention_type": row["convention_type"],
            "original_answer": row["original_answer"],
            "variant_answer": row["variant_answer"],
            "original_em": original_em,
            "variant_em": variant_em,
            "original_f1": original_f1,
            "variant_f1": variant_f1,
            "eligible": eligible,
            "n_gold_annotations": len(gold_answers),
        })
        if eligible:
            for kind, control in control_variants(row["original_answer"]):
                em, f1 = drop_scores(control, gold_answers)
                controls.append({"benchmark": "DROP", "item_id": row["item_id"], "control_type": kind, "em": em, "f1": f1})
    return scored, controls


def score_truthfulqa(records: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation", revision=TRUTHFULQA_REVISION)
    scored = []
    controls = []
    for row in records:
        if row["benchmark"] != "TruthfulQA":
            continue
        index = int(row["item_id"].removeprefix("truthfulqa_"))
        item = dataset[index]
        references = tuple(dict.fromkeys([item["best_answer"], *item["correct_answers"]]))
        original = float(exact_match(row["original_answer"], references))
        variant = float(exact_match(row["variant_answer"], references))
        exclusion_reason = TRUTHFULQA_EXCLUSIONS.get(row["item_id"], "")
        scored.append({
            "benchmark": "TruthfulQA",
            "item_id": row["item_id"],
            "convention_type": row["convention_type"],
            "original_answer": row["original_answer"],
            "variant_answer": row["variant_answer"],
            "original_reference_em": original,
            "variant_reference_em": variant,
            "n_correct_references": len(references),
            "eligible": not exclusion_reason,
            "exclusion_reason": exclusion_reason,
        })
        for kind, control in control_variants(row["original_answer"]) if not exclusion_reason else []:
            controls.append({
                "benchmark": "TruthfulQA",
                "item_id": row["item_id"],
                "control_type": kind,
                "reference_em": float(exact_match(control, references)),
            })
    return scored, controls


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def pp(value: float) -> str:
    return f"{100 * value:.1f}"


def render_report(summary: list[dict[str, object]], drop: list[dict[str, object]], truthfulqa: list[dict[str, object]], controls: list[dict[str, object]], samples: int) -> str:
    eligible = [row for row in drop if row["eligible"]]
    truthfulqa_eligible = [row for row in truthfulqa if row["eligible"]]
    lines = [
        "# Paired scoring with benchmark-specific metrics",
        "",
        "The scorer is applied to both the original reference answer and its convention variant. These are answer strings, not sampled model responses; the results are conditional scoring sensitivity, not observed model accuracy.",
        "",
        f"DROP uses the [AllenNLP official evaluator](https://github.com/allenai/allennlp-models/blob/b1f372248c17ad12684d344955fbcd98e957e77e/allennlp_models/rc/tools/drop.py) and original development annotations. Of {len(drop)} candidate pairs, {len(eligible)} pairs from {len({row['item_id'] for row in eligible})} questions receive full original-answer EM and F1 and enter the paired analysis. The other {len(drop) - len(eligible)} flattened candidates are incomplete gold answers under the official scorer and are excluded. DROP EM and F1 are reported separately.",
        "",
        f"TruthfulQA is a separate normalized exact-reference diagnostic over {len(truthfulqa_eligible)} clean pairs from {len({row['item_id'] for row in truthfulqa_eligible})} questions. Four candidates are excluded: three altered the Flavor-Aid brand and one altered the official name World Health Organization. It matches against any `best_answer` or `correct_answers` alias after lowercasing and removing punctuation, articles, and extra whitespace. It is **not** TruthfulQA's official generation metric.",
        "",
        f"95% percentile intervals resample source questions ({samples:,} replicates); all variants from a sampled question stay together. No pooled statistic combines DROP with TruthfulQA.",
        "",
        "| Benchmark | Metric | Type | Pairs | Questions | Original | Variant | Change [95% CI], pp |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        metric = {"em": "Official EM", "f1": "Official F1", "reference_em": "Normalized reference EM (diagnostic)"}[str(row["metric"])]
        lines.append(f"| {row['benchmark']} | {metric} | {row['group']} | {row['n_pairs']} | {row['n_source_items']} | {pct(float(row['original_score']))} | {pct(float(row['variant_score']))} | {pp(float(row['delta']))} [{pp(float(row['ci_low']))}, {pp(float(row['ci_high']))}] |")
    lines += ["", "## Surface-edit controls", "", "| Benchmark | Controls | Full-score EM | Mean F1 |", "|---|---:|---:|---:|"]
    for benchmark in ("DROP", "TruthfulQA"):
        subset = [row for row in controls if row["benchmark"] == benchmark]
        key = "em" if benchmark == "DROP" else "reference_em"
        f1 = pct(float(np.mean([float(row["f1"]) for row in subset]))) if benchmark == "DROP" else "n/a"
        lines.append(f"| {benchmark} | {len(subset)} | {pct(float(np.mean([float(row[key]) for row in subset])))} | {f1} |")
    lines += [
        "",
        "Controls apply case, whitespace, or trailing-period changes to the same original answers. Their acceptance checks normalization behavior, not semantic equivalence of arbitrary paraphrases.",
        "",
        "## Scoring specification",
        "",
        "For each candidate, we score both the original answer and its convention variant against the same reference set. DROP uses the original development annotations and the AllenNLP evaluator at commit `b1f372248c17ad12684d344955fbcd98e957e77e`; we retain a pair only if its original answer receives both EM = 1 and F1 = 1 under that evaluator. We report the arithmetic mean of paired variant-minus-original EM and F1 changes separately. TruthfulQA uses the generation split at revision `741b8276f2d1982aa3d5b832d3ee81ed3b896490` and a local normalized exact-reference matcher over `best_answer` and `correct_answers`; this is a diagnostic, not the benchmark's official generation metric. We exclude changed proper names and brands from the canonical TruthfulQA slice. For each metric, the 95% percentile interval resamples source questions with replacement, keeping all variants of a sampled question together. We use 10,000 replicates and seed 2026. The estimand is conditional scorer sensitivity on reference-answer variants, not model accuracy or the prevalence of those variants in generated responses.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "results/evaluation_sensitivity_broad/exact_match_variant_records.csv")
    parser.add_argument("--drop-archive", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation_sensitivity_broad")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    records = read_csv(args.records)
    drop, drop_controls = score_drop(records, load_drop_questions(args.drop_archive))
    truthfulqa, truthfulqa_controls = score_truthfulqa(records)
    eligible = [row for row in drop if row["eligible"]]
    if not eligible or not truthfulqa:
        raise ValueError("Missing eligible DROP or TruthfulQA records")
    if any(row["original_reference_em"] != 1.0 for row in truthfulqa):
        raise ValueError("TruthfulQA original answer failed its reference matcher")
    summary = [
        summarize(eligible, "DROP", metric, "all", args.bootstrap_samples, args.seed)
        for metric in ("em", "f1")
    ]
    for kind in sorted({str(row["convention_type"]) for row in eligible}):
        subset = [row for row in eligible if row["convention_type"] == kind]
        summary.extend(summarize(subset, "DROP", metric, kind, args.bootstrap_samples, args.seed) for metric in ("em", "f1"))
    summary.append(summarize([row for row in truthfulqa if row["eligible"]], "TruthfulQA", "reference_em", "all", args.bootstrap_samples, args.seed))
    controls = [*drop_controls, *truthfulqa_controls]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "official_drop_paired_records.csv", drop)
    write_csv(args.output_dir / "truthfulqa_reference_diagnostic_records.csv", truthfulqa)
    write_csv(args.output_dir / "official_drop_and_truthfulqa_controls.csv", controls)
    write_csv(args.output_dir / "paired_benchmark_specific_summary.csv", summary)
    report = render_report(summary, drop, truthfulqa, controls, args.bootstrap_samples)
    (args.output_dir / "PAIRED_BENCHMARK_SPECIFIC_SCORING.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
