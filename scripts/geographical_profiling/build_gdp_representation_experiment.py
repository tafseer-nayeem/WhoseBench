from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pycountry
import requests


ROOT = Path(__file__).resolve().parents[2]
ENGLISH_DIR = ROOT / "outputs" / "tables" / "english_core_suite"
TABLE_DIR = ROOT / "outputs" / "tables" / "gdp_representation"

WORLD_BANK_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
INDICATORS = {
    "gdp_current_usd": "NY.GDP.MKTP.CD",
    "gdp_per_capita_current_usd": "NY.GDP.PCAP.CD",
    "population": "SP.POP.TOTL",
}

BENCHMARKS = [
    ("datacomp_1b_20k", "DataComp1B"),
    ("wit_english_50k", "WIT English"),
    ("redcaps_20k", "RedCaps"),
    ("cc12m_20k", "CC12M"),
    ("coyo_700m_20k", "COYO-700M"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float | int | str | None, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def alpha3_to_alpha2(alpha3: str) -> str | None:
    if not alpha3 or len(alpha3) != 3:
        return None
    country = pycountry.countries.get(alpha_3=alpha3)
    return getattr(country, "alpha_2", None) if country else None


def fetch_world_bank_indicator(indicator: str) -> list[dict[str, object]]:
    url = WORLD_BANK_API.format(indicator=indicator)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise RuntimeError(f"Unexpected World Bank response for {indicator}")
    return payload[1]


def latest_country_values(indicator: str) -> dict[str, dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for row in fetch_world_bank_indicator(indicator):
        value = row.get("value")
        if value is None:
            continue
        alpha2 = alpha3_to_alpha2(str(row.get("countryiso3code", "")))
        if not alpha2:
            continue
        year = int(str(row["date"]))
        existing = latest.get(alpha2)
        if existing is None or year > int(existing["year"]):
            latest[alpha2] = {
                "value": float(value),
                "year": year,
                "world_bank_name": row.get("country", {}).get("value", ""),
            }
    return latest


def build_indicator_table() -> dict[str, dict[str, object]]:
    by_indicator = {
        name: latest_country_values(code)
        for name, code in INDICATORS.items()
    }
    countries = sorted(set().union(*[set(values.keys()) for values in by_indicator.values()]))
    out: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []

    for alpha2 in countries:
        country = pycountry.countries.get(alpha_2=alpha2)
        base: dict[str, object] = {
            "country_alpha2": alpha2,
            "country_alpha3": getattr(country, "alpha_3", ""),
            "country_name": getattr(country, "name", alpha2),
        }
        for name in INDICATORS:
            item = by_indicator[name].get(alpha2, {})
            base[name] = item.get("value")
            base[f"{name}_year"] = item.get("year", "")
        out[alpha2] = base
        rows.append(
            {
                "country_alpha2": base["country_alpha2"],
                "country_alpha3": base["country_alpha3"],
                "country_name": base["country_name"],
                "gdp_current_usd": fmt(base.get("gdp_current_usd"), 2),
                "gdp_current_usd_year": base.get("gdp_current_usd_year", ""),
                "gdp_per_capita_current_usd": fmt(base.get("gdp_per_capita_current_usd"), 2),
                "gdp_per_capita_current_usd_year": base.get("gdp_per_capita_current_usd_year", ""),
                "population": fmt(base.get("population"), 0),
                "population_year": base.get("population_year", ""),
            }
        )

    write_csv(
        TABLE_DIR / "world_bank_country_indicators.csv",
        rows,
        [
            "country_alpha2",
            "country_alpha3",
            "country_name",
            "gdp_current_usd",
            "gdp_current_usd_year",
            "gdp_per_capita_current_usd",
            "gdp_per_capita_current_usd_year",
            "population",
            "population_year",
        ],
    )
    return out


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom == 0.0:
        return None
    return float((x * y).sum() / denom)


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    return pearson(rank_values(xs), rank_values(ys))


def log10(value: float | None) -> float | None:
    if value is None or value <= 0.0 or not math.isfinite(value):
        return None
    return math.log10(value)


def load_representation_rows(indicators: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for benchmark_key, benchmark_name in BENCHMARKS:
        country_rows = read_csv(ENGLISH_DIR / f"{benchmark_key}_country_summary.csv")
        enriched = []
        for row in country_rows:
            alpha2 = row["country_alpha2"]
            econ = indicators.get(alpha2)
            if not econ:
                continue
            gdp = econ.get("gdp_current_usd")
            pop = econ.get("population")
            gdp_pc = econ.get("gdp_per_capita_current_usd")
            if not isinstance(gdp, float) or not isinstance(pop, float) or not isinstance(gdp_pc, float):
                continue
            dataset_share_pct = float(row["share_resolved_pct"])
            count = int(row["count"])
            enriched.append((row, econ, gdp, pop, gdp_pc, dataset_share_pct, count))

        total_gdp = sum(item[2] for item in enriched)
        total_population = sum(item[3] for item in enriched)
        total_dataset_share = sum(item[5] for item in enriched)

        for row, econ, gdp, pop, gdp_pc, dataset_share_pct, count in enriched:
            gdp_share_pct = 100.0 * gdp / total_gdp if total_gdp else 0.0
            population_share_pct = 100.0 * pop / total_population if total_population else 0.0
            dataset_share_normalized_pct = 100.0 * dataset_share_pct / total_dataset_share if total_dataset_share else 0.0
            rep_to_gdp = dataset_share_normalized_pct / gdp_share_pct if gdp_share_pct else None
            rep_to_pop = dataset_share_normalized_pct / population_share_pct if population_share_pct else None
            rows.append(
                {
                    "benchmark_key": benchmark_key,
                    "benchmark_name": benchmark_name,
                    "country_alpha2": row["country_alpha2"],
                    "country_name": row["country_name"],
                    "region": row.get("region", ""),
                    "count": count,
                    "dataset_share_resolved_pct": dataset_share_pct,
                    "dataset_share_normalized_pct": dataset_share_normalized_pct,
                    "gdp_current_usd": gdp,
                    "gdp_year": econ.get("gdp_current_usd_year", ""),
                    "gdp_share_pct": gdp_share_pct,
                    "gdp_per_capita_current_usd": gdp_pc,
                    "gdp_per_capita_year": econ.get("gdp_per_capita_current_usd_year", ""),
                    "population": pop,
                    "population_year": econ.get("population_year", ""),
                    "population_share_pct": population_share_pct,
                    "representation_to_gdp_ratio": rep_to_gdp,
                    "representation_to_population_ratio": rep_to_pop,
                    "log10_dataset_share_normalized_pct": log10(dataset_share_normalized_pct),
                    "log10_gdp_current_usd": log10(gdp),
                    "log10_gdp_per_capita_current_usd": log10(gdp_pc),
                    "log10_population": log10(pop),
                    "log10_representation_to_gdp_ratio": log10(rep_to_gdp),
                    "log10_representation_to_population_ratio": log10(rep_to_pop),
                }
            )

    return rows


def write_long_table(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "benchmark_key",
        "benchmark_name",
        "country_alpha2",
        "country_name",
        "region",
        "count",
        "dataset_share_resolved_pct",
        "dataset_share_normalized_pct",
        "gdp_current_usd",
        "gdp_year",
        "gdp_share_pct",
        "log10_gdp_share_pct",
        "gdp_per_capita_current_usd",
        "gdp_per_capita_year",
        "population",
        "population_year",
        "population_share_pct",
        "representation_to_gdp_ratio",
        "representation_to_population_ratio",
        "log10_dataset_share_normalized_pct",
        "log10_gdp_current_usd",
        "log10_gdp_per_capita_current_usd",
        "log10_population",
        "log10_representation_to_gdp_ratio",
        "log10_representation_to_population_ratio",
    ]
    out_rows = []
    for row in rows:
        out = dict(row)
        for key in fieldnames:
            value = out.get(key)
            if isinstance(value, float):
                out[key] = fmt(value, 6)
        out_rows.append(out)
    write_csv(TABLE_DIR / "gdp_representation_long.csv", out_rows, fieldnames)


def complete_pairs(rows: Iterable[dict[str, object]], x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        x = row.get(x_key)
        y = row.get(y_key)
        if isinstance(x, float) and isinstance(y, float) and math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    return xs, ys


def write_correlation_table(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_benchmark[str(row["benchmark_name"])].append(row)

    specs = [
        ("log10_gdp_current_usd", "log10_dataset_share_normalized_pct", "GDP current USD"),
        ("log10_gdp_per_capita_current_usd", "log10_dataset_share_normalized_pct", "GDP per capita current USD"),
        ("log10_population", "log10_dataset_share_normalized_pct", "Population"),
        ("log10_gdp_current_usd", "log10_representation_to_gdp_ratio", "GDP current USD vs representation/GDP ratio"),
        ("log10_gdp_per_capita_current_usd", "log10_representation_to_gdp_ratio", "GDP per capita vs representation/GDP ratio"),
    ]

    out_rows: list[dict[str, object]] = []
    for benchmark_name, benchmark_rows in by_benchmark.items():
        for x_key, y_key, analysis in specs:
            xs, ys = complete_pairs(benchmark_rows, x_key, y_key)
            out_rows.append(
                {
                    "benchmark_name": benchmark_name,
                    "analysis": analysis,
                    "x_variable": x_key,
                    "y_variable": y_key,
                    "n_countries": len(xs),
                    "pearson_r": fmt(pearson(xs, ys), 4),
                    "spearman_rho": fmt(spearman(xs, ys), 4),
                }
            )

    write_csv(
        TABLE_DIR / "gdp_representation_correlations.csv",
        out_rows,
        [
            "benchmark_name",
            "analysis",
            "x_variable",
            "y_variable",
            "n_countries",
            "pearson_r",
            "spearman_rho",
        ],
    )
    return out_rows


def write_over_under_tables(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_country: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_country[str(row["country_alpha2"])].append(row)

    summary = []
    for alpha2, country_rows in by_country.items():
        ratios = [
            row["log10_representation_to_gdp_ratio"]
            for row in country_rows
            if isinstance(row.get("log10_representation_to_gdp_ratio"), float)
        ]
        if not ratios:
            continue
        country_name = str(country_rows[0]["country_name"])
        summary.append(
            {
                "country_alpha2": alpha2,
                "country_name": country_name,
                "benchmark_presence": len(country_rows),
                "mean_log10_representation_to_gdp_ratio": sum(ratios) / len(ratios),
                "mean_representation_to_gdp_ratio": 10 ** (sum(ratios) / len(ratios)),
                "mean_dataset_share_normalized_pct": sum(float(row["dataset_share_normalized_pct"]) for row in country_rows) / len(country_rows),
                "mean_gdp_share_pct": sum(float(row["gdp_share_pct"]) for row in country_rows) / len(country_rows),
            }
        )

    summary.sort(key=lambda row: float(row["mean_log10_representation_to_gdp_ratio"]), reverse=True)
    out_rows = []
    for row in summary:
        out_rows.append(
            {
                "country_alpha2": row["country_alpha2"],
                "country_name": row["country_name"],
                "benchmark_presence": row["benchmark_presence"],
                "mean_log10_representation_to_gdp_ratio": fmt(float(row["mean_log10_representation_to_gdp_ratio"]), 6),
                "mean_representation_to_gdp_ratio": fmt(float(row["mean_representation_to_gdp_ratio"]), 6),
                "mean_dataset_share_normalized_pct": fmt(float(row["mean_dataset_share_normalized_pct"]), 6),
                "mean_gdp_share_pct": fmt(float(row["mean_gdp_share_pct"]), 6),
            }
        )

    write_csv(
        TABLE_DIR / "gdp_over_under_representation_summary.csv",
        out_rows,
        [
            "country_alpha2",
            "country_name",
            "benchmark_presence",
            "mean_log10_representation_to_gdp_ratio",
            "mean_representation_to_gdp_ratio",
            "mean_dataset_share_normalized_pct",
            "mean_gdp_share_pct",
        ],
    )
    return summary


def add_log_gdp_share_columns(rows: list[dict[str, object]]) -> None:
    for row in rows:
        row["log10_gdp_share_pct"] = log10(float(row["gdp_share_pct"]))


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    indicators = build_indicator_table()
    rows = load_representation_rows(indicators)
    add_log_gdp_share_columns(rows)
    write_long_table(rows)
    correlations = write_correlation_table(rows)
    over_under = write_over_under_tables(rows)
    print(f"Wrote {TABLE_DIR / 'world_bank_country_indicators.csv'}")
    print(f"Wrote {TABLE_DIR / 'gdp_representation_long.csv'}")
    print(f"Wrote {TABLE_DIR / 'gdp_representation_correlations.csv'}")
    print(f"Wrote {TABLE_DIR / 'gdp_over_under_representation_summary.csv'}")


if __name__ == "__main__":
    main()
