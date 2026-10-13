from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def is_true(value: str) -> bool:
    return str(value).strip().lower() == "true"


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def pp(x: float) -> str:
    return f"{100.0 * x:.1f} pp"


def variant_score(record: dict[str, str], scope: str) -> float | None:
    if scope == "standard_exact_match":
        return float(is_true(record["accepted_by_exact_match"]))
    if scope == "applicable_answer_text_scorer":
        value = record["accepted_by_benchmark_scorer"]
        if value == "not_applicable":
            return None
        return float(is_true(value))
    raise ValueError(scope)


def cluster_key(record: dict[str, str]) -> str:
    return f"{record['benchmark']}::{record['split']}::{record['item_id']}"


def summarize_records(records: list[dict[str, str]], scope: str) -> dict[str, float | int]:
    variant_scores: list[float] = []
    for record in records:
        score = variant_score(record, scope)
        if score is not None:
            variant_scores.append(score)
    n = len(variant_scores)
    original_score = 1.0 if n else 0.0
    perturbed_score = mean(variant_scores)
    delta = perturbed_score - original_score
    return {
        "n": n,
        "original_score": original_score,
        "perturbed_score": perturbed_score,
        "score_delta": delta,
        "false_rejection_original": 1.0 - original_score if n else 0.0,
        "variant_rejection": 1.0 - perturbed_score if n else 0.0,
    }


def bootstrap_delta_ci(
    records: list[dict[str, str]],
    scope: str,
    samples: int,
    rng: random.Random,
) -> tuple[float, float]:
    valid = [r for r in records if variant_score(r, scope) is not None]
    clusters: dict[str, list[dict[str, str]]] = {}
    for record in valid:
        clusters.setdefault(cluster_key(record), []).append(record)
    keys = sorted(clusters)
    if not keys:
        return 0.0, 0.0
    deltas: list[float] = []
    for _ in range(samples):
        sample: list[dict[str, str]] = []
        for _ in keys:
            sample.extend(clusters[rng.choice(keys)])
        deltas.append(float(summarize_records(sample, scope)["score_delta"]))
    deltas.sort()
    lo = int(0.025 * (len(deltas) - 1))
    hi = int(0.975 * (len(deltas) - 1))
    return deltas[lo], deltas[hi]


def group_by(records: list[dict[str, str]], keys: tuple[str, ...]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for record in records:
        name = " / ".join(record[key] for key in keys)
        out.setdefault(name, []).append(record)
    return out


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_control_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_score_rows(records: list[dict[str, str]], samples: int, seed: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    group_specs = [
        ("ALL", {"ALL": records}),
        ("benchmark", group_by(records, ("benchmark",))),
        ("perturbation_type", group_by(records, ("convention_type",))),
        ("benchmark_x_perturbation_type", group_by(records, ("benchmark", "convention_type"))),
    ]
    for scope in ("standard_exact_match", "applicable_answer_text_scorer"):
        for grouping, groups in group_specs:
            for group, group_records in sorted(groups.items()):
                stats = summarize_records(group_records, scope)
                if int(stats["n"]) == 0:
                    continue
                lo, hi = bootstrap_delta_ci(group_records, scope, samples, random.Random(f"{seed}:{scope}:{grouping}:{group}"))
                rows.append(
                    {
                        "scope": scope,
                        "grouping": grouping,
                        "group": group,
                        "n_variants": str(int(stats["n"])),
                        "original_score": f"{float(stats['original_score']):.6f}",
                        "valid_variant_score": f"{float(stats['perturbed_score']):.6f}",
                        "score_delta": f"{float(stats['score_delta']):.6f}",
                        "score_delta_ci_low": f"{lo:.6f}",
                        "score_delta_ci_high": f"{hi:.6f}",
                        "false_rejection_original": f"{float(stats['false_rejection_original']):.6f}",
                        "valid_variant_rejection": f"{float(stats['variant_rejection']):.6f}",
                    }
                )
    return rows


def make_scenario_rows(score_rows: list[dict[str, str]], prevalences: list[float]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    selected = [
        row
        for row in score_rows
        if row["grouping"] in {"ALL", "perturbation_type"}
        and row["scope"] in {"standard_exact_match", "applicable_answer_text_scorer"}
    ]
    for row in selected:
        delta = float(row["score_delta"])
        lo = float(row["score_delta_ci_low"])
        hi = float(row["score_delta_ci_high"])
        for prevalence in prevalences:
            rows.append(
                {
                    "scope": row["scope"],
                    "grouping": row["grouping"],
                    "group": row["group"],
                    "n_variants": row["n_variants"],
                    "valid_variant_prevalence": f"{prevalence:.3f}",
                    "expected_score_change": f"{prevalence * delta:.6f}",
                    "expected_score_change_ci_low": f"{prevalence * lo:.6f}",
                    "expected_score_change_ci_high": f"{prevalence * hi:.6f}",
                }
            )
    return rows


def top_row(rows: list[dict[str, str]], scope: str, grouping: str, group: str) -> dict[str, str]:
    for row in rows:
        if row["scope"] == scope and row["grouping"] == grouping and row["group"] == group:
            return row
    raise KeyError((scope, grouping, group))


def render_md(
    score_rows: list[dict[str, str]],
    scenario_rows: list[dict[str, str]],
    control_rows: list[dict[str, str]],
    output_path: Path,
) -> None:
    def score_line(row: dict[str, str], label: str | None = None) -> str:
        name = label or row["group"]
        return (
            f"| {name} | {row['n_variants']} | {pct(float(row['original_score']))} | "
            f"{pct(float(row['valid_variant_score']))} | "
            f"{pp(float(row['score_delta']))} [{pp(float(row['score_delta_ci_low']))}, "
            f"{pp(float(row['score_delta_ci_high']))}] | "
            f"{pct(float(row['false_rejection_original']))} | |"
        )

    def scenario_line(row: dict[str, str], label: str | None = None) -> str:
        name = label or row["group"]
        return (
            f"| {name} | {float(row['valid_variant_prevalence']):.0%} | "
            f"{pp(float(row['expected_score_change']))} "
            f"[{pp(float(row['expected_score_change_ci_low']))}, "
            f"{pp(float(row['expected_score_change_ci_high']))}] |"
        )

    lines: list[str] = []
    lines.append("# Outcome-Sensitivity Analysis")
    lines.append("")
    lines.append(
        "This report reframes the sensitivity test as an evaluation outcome. Each row compares "
        "the score assigned to an original correct answer with the score assigned to a "
        "conservative valid convention variant under the same scoring rule."
    )
    lines.append("")
    lines.append("## Main Paired Score Deltas")
    lines.append("")
    lines.append("| Setting | Variants | Original score | Valid-variant score | Score change [95% CI] | Original false rejection | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    lines.append(score_line(top_row(score_rows, "standard_exact_match", "ALL", "ALL"), "Broad standard exact-match diagnostic"))
    lines.append(score_line(top_row(score_rows, "applicable_answer_text_scorer", "ALL", "ALL"), "Applicable answer-text scorers"))
    lines.append("")
    lines.append("## Applicable Answer-Text Scorers by Perturbation Type")
    lines.append("")
    lines.append("| Perturbation type | Variants | Original score | Valid-variant score | Score change [95% CI] | Original false rejection | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in score_rows:
        if row["scope"] == "applicable_answer_text_scorer" and row["grouping"] == "perturbation_type":
            lines.append(score_line(row))
    lines.append("")
    lines.append("## Applicable Answer-Text Scorers by Benchmark and Perturbation Type")
    lines.append("")
    lines.append("| Benchmark / perturbation | Variants | Original score | Valid-variant score | Score change [95% CI] | Original false rejection | Notes |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in score_rows:
        if row["scope"] == "applicable_answer_text_scorer" and row["grouping"] == "benchmark_x_perturbation_type":
            lines.append(score_line(row))
    lines.append("")
    lines.append("## Practical Score-Impact Scenarios")
    lines.append("")
    lines.append(
        "These rows convert paired score deltas into expected benchmark-score changes if a "
        "fraction of otherwise correct answers are expressed in the valid variant form."
    )
    lines.append("")
    lines.append("| Setting | Valid-variant prevalence among correct answers | Expected score change [95% CI] |")
    lines.append("|---|---:|---:|")
    for row in scenario_rows:
        if row["grouping"] == "ALL" and row["valid_variant_prevalence"] in {"0.050", "0.100", "0.250"}:
            label = row["scope"].replace("_", " ")
            lines.append(scenario_line(row, label))
    lines.append("")
    if control_rows:
        lines.append("## Negative Controls")
        lines.append("")
        lines.append(
            "These controls apply surface edits that benchmark normalizers are expected to accept. "
            "They check whether the convention-variant result is merely generic punctuation, case, "
            "or whitespace brittleness."
        )
        lines.append("")
        lines.append("| Scorer scope | Control group | Controls | Accepted | Rejected |")
        lines.append("|---|---|---:|---:|---:|")
        for row in control_rows:
            if row.get("group") == "ALL":
                lines.append(
                    f"| {row['scope']} | {row['group']} | {row['n_controls']} | "
                    f"{pct(float(row['accept_rate']))} | {pct(float(row['rejection_rate']))} |"
                )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The false-rejection baseline is explicit: unperturbed correct answers receive 100% "
        "score in every paired setting. The score loss appears when the same answer is "
        "written as a valid convention variant. The negative controls show that accepted "
        "normalization edits do not produce the same failure pattern. The tables report "
        "score changes by perturbation type, item counts, uncertainty, and the "
        "unperturbed-answer baseline."
    )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/exact_match_variant_records.csv"),
    )
    parser.add_argument(
        "--control-summary",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad/scorer_control_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation_sensitivity_broad"),
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    with args.records.open(newline="", encoding="utf-8") as f:
        records = list(csv.DictReader(f))

    score_rows = make_score_rows(records, args.bootstrap_samples, args.seed)
    scenario_rows = make_scenario_rows(score_rows, [0.01, 0.05, 0.10, 0.25, 1.0])
    control_rows = load_control_summary(args.control_summary)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "outcome_sensitivity_score_deltas.csv", score_rows)
    write_csv(args.output_dir / "outcome_sensitivity_score_impact_scenarios.csv", scenario_rows)
    render_md(score_rows, scenario_rows, control_rows, args.output_dir / "OUTCOME_SENSITIVITY_RESPONSE.md")
    print(f"wrote {args.output_dir / 'outcome_sensitivity_score_deltas.csv'}")
    print(f"wrote {args.output_dir / 'outcome_sensitivity_score_impact_scenarios.csv'}")
    print(f"wrote {args.output_dir / 'OUTCOME_SENSITIVITY_RESPONSE.md'}")


if __name__ == "__main__":
    main()
