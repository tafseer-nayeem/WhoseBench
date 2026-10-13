from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pycountry
import requests


ROOT = Path(__file__).resolve().parents[2]
BASE_TABLE_DIR = ROOT / "outputs" / "tables" / "gdp_representation"
TABLE_DIR = ROOT / "outputs" / "tables" / "gdp_representation_v2"

WORLD_BANK_API = "https://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"

OPTIONAL_WORLD_BANK_INDICATORS = {
    "internet_users_pct": "IT.NET.USER.ZS",
    "tourism_arrivals": "ST.INT.ARVL",
}

BENCHMARKS = [
    ("datacomp_1b_20k", "DataComp1B"),
    ("wit_english_50k", "WIT English"),
    ("redcaps_20k", "RedCaps"),
    ("cc12m_20k", "CC12M"),
    ("coyo_700m_20k", "COYO-700M"),
]


RECOMMENDED_SETTING_ID = "pop_ge_1m"


@dataclass(frozen=True)
class Setting:
    setting_id: str
    label: str
    min_count: int = 1
    min_population: int | None = None


SETTINGS = [
    Setting("all_countries", "All countries", min_count=1),
    Setting("pop_ge_500k", "Population >= 500k", min_count=1, min_population=500_000),
    Setting("pop_ge_1m", "Population >= 1M", min_count=1, min_population=1_000_000),
    Setting("pop_ge_2m", "Population >= 2M", min_count=1, min_population=2_000_000),
    Setting("min3_all", "Mentions >= 3", min_count=3),
    Setting("min3_pop_ge_1m", "Mentions >= 3 + population >= 1M", min_count=3, min_population=1_000_000),
    Setting("min5_all", "Mentions >= 5", min_count=5),
    Setting("min5_pop_ge_1m", "Mentions >= 5 + population >= 1M", min_count=5, min_population=1_000_000),
    Setting("min10_pop_ge_1m", "Mentions >= 10 + population >= 1M", min_count=10, min_population=1_000_000),
]


MODEL_SPECS = [
    ("gdp_share", "GDP share only", ["log10_gdp_share_pct"]),
    ("population_share", "Population share only", ["log10_population_share_pct"]),
    ("gdp_per_capita", "GDP per capita only", ["log10_gdp_per_capita_current_usd"]),
    ("gdp_plus_population", "GDP share + population share", ["log10_gdp_share_pct", "log10_population_share_pct"]),
    ("gdp_plus_gdp_pc", "GDP share + GDP per capita", ["log10_gdp_share_pct", "log10_gdp_per_capita_current_usd"]),
    ("gdp_plus_internet", "GDP share + internet users %", ["log10_gdp_share_pct", "log10_internet_users_pct"]),
    ("gdp_plus_tourism", "GDP share + tourism arrivals", ["log10_gdp_share_pct", "log10_tourism_arrivals"]),
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


def fmt(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}f}"
    return str(value)


def parse_number(value: str) -> float | int | None:
    if value == "":
        return None
    try:
        as_float = float(value)
    except ValueError:
        return None
    if as_float.is_integer():
        return int(as_float)
    return as_float


def log10(value: float | int | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0.0 or not math.isfinite(value):
        return None
    return math.log10(value)


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


def build_optional_indicator_table() -> dict[str, dict[str, object]]:
    by_indicator = {
        name: latest_country_values(code)
        for name, code in OPTIONAL_WORLD_BANK_INDICATORS.items()
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
        for name in OPTIONAL_WORLD_BANK_INDICATORS:
            item = by_indicator[name].get(alpha2, {})
            base[name] = item.get("value")
            base[f"{name}_year"] = item.get("year", "")
        out[alpha2] = base
        rows.append(
            {
                "country_alpha2": base["country_alpha2"],
                "country_alpha3": base["country_alpha3"],
                "country_name": base["country_name"],
                "internet_users_pct": fmt(base.get("internet_users_pct"), 6),
                "internet_users_pct_year": base.get("internet_users_pct_year", ""),
                "tourism_arrivals": fmt(base.get("tourism_arrivals"), 0),
                "tourism_arrivals_year": base.get("tourism_arrivals_year", ""),
            }
        )

    write_csv(
        TABLE_DIR / "world_bank_country_controls.csv",
        rows,
        [
            "country_alpha2",
            "country_alpha3",
            "country_name",
            "internet_users_pct",
            "internet_users_pct_year",
            "tourism_arrivals",
            "tourism_arrivals_year",
        ],
    )
    return out


def load_base_rows() -> list[dict[str, object]]:
    path = BASE_TABLE_DIR / "gdp_representation_long.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run scripts/build_gdp_representation_experiment.py first."
        )
    rows: list[dict[str, object]] = []
    string_fields = {"benchmark_key", "benchmark_name", "country_alpha2", "country_name", "region"}
    for raw in read_csv(path):
        row: dict[str, object] = {}
        for key, value in raw.items():
            if key in string_fields:
                row[key] = value
            elif key == "count":
                parsed = parse_number(value)
                row[key] = int(parsed) if parsed is not None else None
            else:
                row[key] = parse_number(value)
        row["log10_population_share_pct"] = log10(row.get("population_share_pct"))
        rows.append(row)
    return rows


def enrich_with_optional_controls(rows: list[dict[str, object]], controls: dict[str, dict[str, object]]) -> None:
    for row in rows:
        control = controls.get(str(row["country_alpha2"]), {})
        internet = control.get("internet_users_pct")
        tourism = control.get("tourism_arrivals")
        row["internet_users_pct"] = internet if isinstance(internet, float) else None
        row["internet_users_pct_year"] = control.get("internet_users_pct_year", "")
        row["tourism_arrivals"] = tourism if isinstance(tourism, float) else None
        row["tourism_arrivals_year"] = control.get("tourism_arrivals_year", "")
        row["log10_internet_users_pct"] = log10(row["internet_users_pct"])
        row["log10_tourism_arrivals"] = log10(row["tourism_arrivals"])


def write_v2_long_table(rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "benchmark_key",
        "benchmark_name",
        "country_alpha2",
        "country_name",
        "region",
        "count",
        "dataset_share_normalized_pct",
        "gdp_share_pct",
        "population_share_pct",
        "gdp_current_usd",
        "gdp_year",
        "gdp_per_capita_current_usd",
        "gdp_per_capita_year",
        "population",
        "population_year",
        "internet_users_pct",
        "internet_users_pct_year",
        "tourism_arrivals",
        "tourism_arrivals_year",
        "representation_to_gdp_ratio",
        "log10_dataset_share_normalized_pct",
        "log10_gdp_share_pct",
        "log10_gdp_current_usd",
        "log10_population_share_pct",
        "log10_population",
        "log10_gdp_per_capita_current_usd",
        "log10_internet_users_pct",
        "log10_tourism_arrivals",
        "log10_representation_to_gdp_ratio",
    ]
    out_rows = []
    for row in rows:
        out = {}
        for key in fieldnames:
            out[key] = fmt(row.get(key), 6)
        out_rows.append(out)
    write_csv(TABLE_DIR / "gdp_representation_v2_long.csv", out_rows, fieldnames)


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


def weighted_pearson(xs: list[float], ys: list[float], weights: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys) or len(xs) != len(weights):
        return None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    w = np.asarray(weights, dtype=float)
    if np.any(w <= 0.0) or float(w.sum()) == 0.0:
        return None
    w = w / w.sum()
    mx = float((w * x).sum())
    my = float((w * y).sum())
    dx = x - mx
    dy = y - my
    denom = float(np.sqrt((w * dx * dx).sum() * (w * dy * dy).sum()))
    if denom == 0.0:
        return None
    return float((w * dx * dy).sum() / denom)


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


def complete_rows(rows: Iterable[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    complete = []
    for row in rows:
        ok = True
        for key in keys:
            value = row.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                ok = False
                break
        if ok:
            complete.append(row)
    return complete


def filter_rows(rows: Iterable[dict[str, object]], setting: Setting) -> list[dict[str, object]]:
    filtered = []
    for row in rows:
        count = row.get("count")
        population = row.get("population")
        if not isinstance(count, int) or count < setting.min_count:
            continue
        if setting.min_population is not None:
            if not isinstance(population, (float, int)) or float(population) < setting.min_population:
                continue
        filtered.append(row)
    return filtered


def simple_regression(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
    if len(xs) < 3:
        return None, None
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if float(np.var(x)) == 0.0:
        return None, None
    slope, intercept = np.polyfit(x, y, deg=1)
    y_hat = slope * x + intercept
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), r2


def ols_r2(rows: list[dict[str, object]], x_keys: list[str], y_key: str, weighted: bool = False) -> float | None:
    complete = complete_rows(rows, x_keys + [y_key, "count"])
    if len(complete) < len(x_keys) + 3:
        return None

    y = np.asarray([float(row[y_key]) for row in complete], dtype=float)
    x_cols = [np.ones(len(complete))]
    for key in x_keys:
        x_cols.append(np.asarray([float(row[key]) for row in complete], dtype=float))
    x = np.column_stack(x_cols)

    if weighted:
        weights = np.asarray([float(row["count"]) for row in complete], dtype=float)
        if np.any(weights <= 0.0):
            return None
        sqrt_w = np.sqrt(weights)
        x_fit = x * sqrt_w[:, None]
        y_fit = y * sqrt_w
    else:
        weights = None
        x_fit = x
        y_fit = y

    try:
        beta, *_ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    except np.linalg.LinAlgError:
        return None

    y_hat = x @ beta
    if weighted:
        w = weights / weights.sum()
        y_mean = float((w * y).sum())
        ss_res = float((w * (y - y_hat) ** 2).sum())
        ss_tot = float((w * (y - y_mean) ** 2).sum())
    else:
        y_mean = float(y.mean())
        ss_res = float(((y - y_hat) ** 2).sum())
        ss_tot = float(((y - y_mean) ** 2).sum())
    if ss_tot == 0.0:
        return None
    return float(1.0 - ss_res / ss_tot)


def setting_statistics(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    detail_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_benchmark[str(row["benchmark_name"])].append(row)

    for setting in SETTINGS:
        valid_details = []
        for _, benchmark_name in BENCHMARKS:
            benchmark_rows = filter_rows(by_benchmark[benchmark_name], setting)
            benchmark_rows = complete_rows(
                benchmark_rows,
                ["log10_gdp_share_pct", "log10_dataset_share_normalized_pct", "count"],
            )
            xs = [float(row["log10_gdp_share_pct"]) for row in benchmark_rows]
            ys = [float(row["log10_dataset_share_normalized_pct"]) for row in benchmark_rows]
            weights = [float(row["count"]) for row in benchmark_rows]
            slope, r2 = simple_regression(xs, ys)
            row = {
                "setting_id": setting.setting_id,
                "setting_label": setting.label,
                "min_count": setting.min_count,
                "min_population": setting.min_population or "",
                "benchmark_name": benchmark_name,
                "n_countries": len(xs),
                "total_country_mentions": int(sum(weights)),
                "pearson_r": pearson(xs, ys),
                "spearman_rho": spearman(xs, ys),
                "weighted_pearson_r": weighted_pearson(xs, ys, weights),
                "ols_slope": slope,
                "ols_r2": r2,
            }
            detail_rows.append(row)
            if row["pearson_r"] is not None:
                valid_details.append(row)

        summary_rows.append(summarize_setting(setting, valid_details))

    write_csv(
        TABLE_DIR / "gdp_representation_v2_setting_detail.csv",
        format_rows(detail_rows),
        [
            "setting_id",
            "setting_label",
            "min_count",
            "min_population",
            "benchmark_name",
            "n_countries",
            "total_country_mentions",
            "pearson_r",
            "spearman_rho",
            "weighted_pearson_r",
            "ols_slope",
            "ols_r2",
        ],
    )
    write_csv(
        TABLE_DIR / "gdp_representation_v2_setting_summary.csv",
        format_rows(summary_rows),
        [
            "setting_id",
            "setting_label",
            "min_count",
            "min_population",
            "valid_benchmarks",
            "total_countries",
            "mean_pearson_r",
            "mean_spearman_rho",
            "mean_weighted_pearson_r",
            "mean_ols_r2",
        ],
    )
    return detail_rows, summary_rows


def summarize_setting(setting: Setting, detail_rows: list[dict[str, object]]) -> dict[str, object]:
    def mean_key(key: str) -> float | None:
        values = [float(row[key]) for row in detail_rows if isinstance(row.get(key), float)]
        if not values:
            return None
        return sum(values) / len(values)

    return {
        "setting_id": setting.setting_id,
        "setting_label": setting.label,
        "min_count": setting.min_count,
        "min_population": setting.min_population or "",
        "valid_benchmarks": len(detail_rows),
        "total_countries": sum(int(row["n_countries"]) for row in detail_rows),
        "mean_pearson_r": mean_key("pearson_r"),
        "mean_spearman_rho": mean_key("spearman_rho"),
        "mean_weighted_pearson_r": mean_key("weighted_pearson_r"),
        "mean_ols_r2": mean_key("ols_r2"),
    }


def regression_model_table(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    setting = next(item for item in SETTINGS if item.setting_id == RECOMMENDED_SETTING_ID)
    by_benchmark: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in filter_rows(rows, setting):
        by_benchmark[str(row["benchmark_name"])].append(row)

    detail_rows: list[dict[str, object]] = []
    for model_id, model_label, x_keys in MODEL_SPECS:
        for _, benchmark_name in BENCHMARKS:
            benchmark_rows = complete_rows(
                by_benchmark[benchmark_name],
                x_keys + ["log10_dataset_share_normalized_pct", "count"],
            )
            detail_rows.append(
                {
                    "setting_id": setting.setting_id,
                    "setting_label": setting.label,
                    "model_id": model_id,
                    "model_label": model_label,
                    "benchmark_name": benchmark_name,
                    "x_variables": " + ".join(x_keys),
                    "n_countries": len(benchmark_rows),
                    "ols_r2": ols_r2(benchmark_rows, x_keys, "log10_dataset_share_normalized_pct", weighted=False),
                    "weighted_ols_r2": ols_r2(benchmark_rows, x_keys, "log10_dataset_share_normalized_pct", weighted=True),
                }
            )

    summary_rows: list[dict[str, object]] = []
    by_model: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in detail_rows:
        by_model[str(row["model_id"])].append(row)
    for model_id, model_label, x_keys in MODEL_SPECS:
        model_rows = [
            row for row in by_model[model_id]
            if isinstance(row.get("ols_r2"), float)
        ]
        weighted_rows = [
            row for row in by_model[model_id]
            if isinstance(row.get("weighted_ols_r2"), float)
        ]
        summary_rows.append(
            {
                "setting_id": setting.setting_id,
                "setting_label": setting.label,
                "model_id": model_id,
                "model_label": model_label,
                "x_variables": " + ".join(x_keys),
                "valid_benchmarks": len(model_rows),
                "min_n_countries": min((int(row["n_countries"]) for row in model_rows), default=""),
                "mean_ols_r2": mean([float(row["ols_r2"]) for row in model_rows]),
                "mean_weighted_ols_r2": mean([float(row["weighted_ols_r2"]) for row in weighted_rows]),
            }
        )

    write_csv(
        TABLE_DIR / "gdp_representation_v2_model_detail.csv",
        format_rows(detail_rows),
        [
            "setting_id",
            "setting_label",
            "model_id",
            "model_label",
            "benchmark_name",
            "x_variables",
            "n_countries",
            "ols_r2",
            "weighted_ols_r2",
        ],
    )
    write_csv(
        TABLE_DIR / "gdp_representation_v2_model_summary.csv",
        format_rows(summary_rows),
        [
            "setting_id",
            "setting_label",
            "model_id",
            "model_label",
            "x_variables",
            "valid_benchmarks",
            "min_n_countries",
            "mean_ols_r2",
            "mean_weighted_ols_r2",
        ],
    )
    return detail_rows, summary_rows


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out_rows = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if isinstance(value, float):
                out[key] = fmt(value, 6)
            else:
                out[key] = "" if value is None else value
        out_rows.append(out)
    return out_rows


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_base_rows()
    controls = build_optional_indicator_table()
    enrich_with_optional_controls(rows, controls)
    write_v2_long_table(rows)
    detail_rows, summary_rows = setting_statistics(rows)
    _, model_summary = regression_model_table(rows)
    print(f"Wrote tables to {TABLE_DIR}")


if __name__ == "__main__":
    main()
