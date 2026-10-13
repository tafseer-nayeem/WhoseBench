#!/usr/bin/env python3
"""Consistency checks for included aggregate outputs."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_rows(rel_path: str) -> list[dict[str, str]]:
    path = ROOT / rel_path
    if not path.exists():
        raise AssertionError(f"Missing expected file: {rel_path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(actual: float, expected: float, tol: float = 1e-3, label: str = "value") -> None:
    if math.isnan(actual) or abs(actual - expected) > tol:
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def check_documentation_audit() -> None:
    summary = read_rows("data/documentation_audit/documentation_audit_summary.csv")
    if len(summary) != 6:
        raise AssertionError(f"documentation summary should have 6 criteria, got {len(summary)}")

    by_id = {row["criterion_id"]: row for row in summary}
    assert_close(float(by_id["source_user_context"]["mean_score"]), 2.00, label="source mean")
    assert_close(float(by_id["target_user_context"]["mean_score"]), 1.00, label="target mean")
    assert_close(float(by_id["geographic_metadata"]["mean_score"]), 0.69, tol=0.01, label="geo mean")

    agreement = read_rows("data/documentation_audit/documentation_audit_agreement_summary.csv")
    overall = next(row for row in agreement if row["criterion_id"] == "overall_micro")
    assert_close(float(overall["exact_agreement_pct"]), 99.0, label="overall exact agreement")
    assert_close(float(overall["cohen_kappa"]), 0.984, label="overall kappa")


def check_convention_outputs() -> None:
    panel = read_rows("results/convention/tables/convention_summary/category_panel_values.csv")
    labels = {row["label"] for row in panel}
    expected = {
        "MMLU",
        "GSM8K",
        "HellaSwag",
        "DROP",
        "WinoGrande",
        "ARC-C",
        "BoolQ",
        "MMLU-Pro",
        "TruthfulQA",
        "MedQA",
        "SVAMP",
    }
    missing = expected - labels
    if missing:
        raise AssertionError(f"missing convention panel rows: {sorted(missing)}")

    mmlu = next(row for row in panel if row["label"] == "MMLU")
    assert_close(float(mmlu["Spelling total"]), 71.81, tol=0.01, label="MMLU spelling total")


def check_geo_outputs() -> None:
    settings = read_rows(
        "results/geographical_profiling/tables/gdp_representation_v2/"
        "gdp_representation_v2_setting_summary.csv"
    )
    pop1m = next(row for row in settings if row["setting_id"] == "pop_ge_1m")
    assert_close(float(pop1m["mean_pearson_r"]), 0.780338, label="GDP mean Pearson")
    assert_close(float(pop1m["mean_spearman_rho"]), 0.778400, label="GDP mean Spearman")
    assert_close(float(pop1m["mean_ols_r2"]), 0.614009, label="GDP mean R2")

    models = read_rows(
        "results/geographical_profiling/tables/gdp_representation_v2/"
        "gdp_representation_v2_model_summary.csv"
    )
    model_ids = {row["model_id"] for row in models}
    for model_id in {"gdp_share", "gdp_plus_internet", "gdp_plus_tourism"}:
        if model_id not in model_ids:
            raise AssertionError(f"missing GDP-control model: {model_id}")


def check_historical_exact_match_scan() -> None:
    summary = read_rows("results/evaluation_sensitivity_broad/exact_match_variant_summary.csv")
    all_row = next(
        row for row in summary if row["benchmark"] == "ALL" and row["convention_type"] == "ALL"
    )
    if int(all_row["num_variant_records"]) != 1726:
        raise AssertionError(
            f"exact-match sensitivity variants: expected 1726, got {all_row['num_variant_records']}"
        )
    if int(all_row["num_source_items"]) != 1619:
        raise AssertionError(
            f"exact-match sensitivity source items: expected 1619, got {all_row['num_source_items']}"
        )
    assert_close(
        float(all_row["exact_match_rejection_rate"]),
        0.997954,
        label="exact-match sensitivity rejection rate",
    )

    truthfulqa = next(
        row for row in summary if row["benchmark"] == "TruthfulQA" and row["convention_type"] == "ALL"
    )
    assert_close(
        float(truthfulqa["benchmark_scorer_rejection_rate_when_applicable"]),
        1.0,
        label="TruthfulQA exact-match sensitivity",
    )

    drop = next(
        row for row in summary if row["benchmark"] == "DROP" and row["convention_type"] == "ALL"
    )
    assert_close(
        float(drop["benchmark_scorer_rejection_rate_when_applicable"]),
        0.521739,
        label="DROP historical proxy sensitivity",
    )

    outcome = read_rows("results/evaluation_sensitivity_broad/outcome_sensitivity_score_deltas.csv")
    applicable = next(
        row
        for row in outcome
        if row["scope"] == "applicable_answer_text_scorer"
        and row["grouping"] == "ALL"
        and row["group"] == "ALL"
    )
    if int(applicable["n_variants"]) != 60:
        raise AssertionError(
            f"applicable scorer outcome variants: expected 60, got {applicable['n_variants']}"
        )
    assert_close(
        float(applicable["score_delta"]),
        -0.816667,
        label="historical mixed-proxy score delta",
    )

    scenarios = read_rows("results/evaluation_sensitivity_broad/outcome_sensitivity_score_impact_scenarios.csv")
    applicable_10pct = next(
        row
        for row in scenarios
        if row["scope"] == "applicable_answer_text_scorer"
        and row["grouping"] == "ALL"
        and row["group"] == "ALL"
        and row["valid_variant_prevalence"] == "0.100"
    )
    assert_close(
        float(applicable_10pct["expected_score_change"]),
        -0.081667,
        label="historical mixed-proxy score impact",
    )

    controls = read_rows("results/evaluation_sensitivity_broad/scorer_control_summary.csv")
    applicable_controls = next(
        row
        for row in controls
        if row["scope"] == "applicable_answer_text_scorer" and row["group"] == "ALL"
    )
    assert_close(
        float(applicable_controls["accept_rate"]),
        1.0,
        label="historical mixed-proxy control accept rate",
    )


def check_benchmark_specific_scoring() -> None:
    drop = read_rows("results/evaluation_sensitivity_broad/official_drop_paired_records.csv")
    eligible = [row for row in drop if row["eligible"] == "True"]
    if len(drop) != 23 or len(eligible) != 12:
        raise AssertionError(f"DROP candidates/eligible pairs: expected 23/12, got {len(drop)}/{len(eligible)}")
    if len({row["item_id"] for row in eligible}) != 11:
        raise AssertionError("DROP eligible pairs should come from 11 source questions")

    truthfulqa = read_rows("results/evaluation_sensitivity_broad/truthfulqa_reference_diagnostic_records.csv")
    if len(truthfulqa) != 37 or len({row["item_id"] for row in truthfulqa}) != 27:
        raise AssertionError("TruthfulQA candidate audit should contain 37 pairs from 27 questions")
    clean_truthfulqa = [row for row in truthfulqa if row["eligible"] == "True"]
    if len(clean_truthfulqa) != 33 or len({row["item_id"] for row in clean_truthfulqa}) != 25:
        raise AssertionError("TruthfulQA clean slice should contain 33 pairs from 25 questions")

    summary = read_rows("results/evaluation_sensitivity_broad/paired_benchmark_specific_summary.csv")
    by_key = {(row["benchmark"], row["metric"], row["group"]): row for row in summary}
    for key, pairs, items, variant, delta in [
        (("DROP", "em", "all"), 12, 11, 1 / 3, -2 / 3),
        (("DROP", "f1", "all"), 12, 11, 0.7716666667, -0.2283333333),
        (("TruthfulQA", "reference_em", "all"), 33, 25, 0.0, -1.0),
    ]:
        row = by_key[key]
        if int(row["n_pairs"]) != pairs or int(row["n_source_items"]) != items:
            raise AssertionError(f"{key}: incorrect pair/question denominators")
        assert_close(float(row["original_score"]), 1.0, label=f"{key} original score")
        assert_close(float(row["variant_score"]), variant, label=f"{key} variant score")
        assert_close(float(row["delta"]), delta, label=f"{key} score change")
        if not float(row["ci_low"]) <= float(row["delta"]) <= float(row["ci_high"]):
            raise AssertionError(f"{key}: point estimate lies outside its confidence interval")


def check_truthfulqa_generation_similarity() -> None:
    records = read_rows("results/evaluation_sensitivity_broad/truthfulqa_generation_similarity_records.csv")
    if len(records) != 33 or len({row["item_id"] for row in records}) != 25:
        raise AssertionError("TruthfulQA generation-similarity slice should contain 33 pairs from 25 questions")
    summary = read_rows("results/evaluation_sensitivity_broad/truthfulqa_generation_similarity_summary.csv")
    by_key = {(row["metric"], row["convention_type"]): row for row in summary}
    for metric, original, variant in [
        ("bleu_acc", 32 / 33, 32 / 33),
        ("rouge1_acc", 32 / 33, 32 / 33),
        ("rouge2_acc", 1.0, 31 / 33),
        ("rougeLsum_acc", 1.0, 1.0),
        ("rouge1_max", 100.0, 88.9352392331),
    ]:
        row = by_key[(metric, "all")]
        if int(row["n_pairs"]) != 33 or int(row["n_source_items"]) != 25:
            raise AssertionError(f"{metric}: incorrect denominators")
        assert_close(float(row["original_score"]), original, label=f"{metric} original")
        assert_close(float(row["variant_score"]), variant, label=f"{metric} variant")
        if not float(row["ci_low"]) <= float(row["delta"]) <= float(row["ci_high"]):
            raise AssertionError(f"{metric}: point estimate lies outside its confidence interval")


def main() -> int:
    checks = [
        check_documentation_audit,
        check_convention_outputs,
        check_geo_outputs,
        check_historical_exact_match_scan,
        check_benchmark_specific_scoring,
        check_truthfulqa_generation_similarity,
    ]
    try:
        for check in checks:
            check()
    except Exception as exc:
        print(f"Output check failed: {exc}")
        return 1
    print("All output checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
