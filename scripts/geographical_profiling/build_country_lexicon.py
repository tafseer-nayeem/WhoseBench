from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pycountry
from babel import Locale
import geonamescache


LOCALES = ["en", "fr", "de", "es", "pt", "ja", "hi", "el", "zh_Hans", "zh_Hant"]
MANUAL_ALIASES = {
    "US": ["united states", "united states of america", "u.s.", "u.s.a.", "usa", "america"],
    "GB": ["united kingdom", "u.k.", "uk", "britain", "great britain"],
    "AE": ["uae", "u.a.e.", "united arab emirates"],
    "KR": ["south korea", "republic of korea"],
    "KP": ["north korea", "dprk", "democratic people's republic of korea"],
    "CZ": ["czech republic", "czechia"],
    "RU": ["russia", "russian federation"],
    "VN": ["vietnam", "viet nam"],
    "IR": ["iran", "iran, islamic republic of"],
    "SY": ["syria", "syrian arab republic"],
    "TZ": ["tanzania", "united republic of tanzania"],
    "BO": ["bolivia", "bolivia, plurinational state of"],
    "VE": ["venezuela", "venezuela, bolivarian republic of"],
    "MD": ["moldova", "republic of moldova"],
    "LA": ["laos", "lao pdr", "lao people's democratic republic"],
    "PS": ["palestine", "state of palestine"],
    "TW": ["taiwan", "taiwan, province of china"],
}


MANUAL_DEMONYMS = {
    "GB": ["british"],
    "CA": ["canadian"],
    "FR": ["french"],
    "DE": ["german"],
    "JP": ["japanese"],
    "CN": ["chinese"],
    "IT": ["italian"],
    "ES": ["spanish"],
    "PT": ["portuguese"],
    "AR": ["argentine", "argentinian"],
    "BR": ["brazilian"],
    "MX": ["mexican"],
    "IN": ["indian"],
    "AU": ["australian"],
    "RU": ["russian"],
    "UA": ["ukrainian"],
    "IL": ["israeli"],
    "PS": ["palestinian"],
    "PL": ["polish"],
    "CZ": ["czech"],
    "SK": ["slovak"],
    "RO": ["romanian"],
    "HU": ["hungarian"],
    "GR": ["greek"],
    "TR": ["turkish"],
    "SE": ["swedish"],
    "NO": ["norwegian"],
    "DK": ["danish"],
    "FI": ["finnish"],
    "IE": ["irish"],
    "NL": ["dutch"],
    "BE": ["belgian"],
    "CH": ["swiss"],
    "AT": ["austrian"],
    "HR": ["croatian"],
    "RS": ["serbian"],
    "SI": ["slovenian"],
    "LT": ["lithuanian"],
    "LV": ["latvian"],
    "EE": ["estonian"],
    "IS": ["icelandic"],
    "TH": ["thai"],
    "VN": ["vietnamese"],
    "ID": ["indonesian"],
    "PH": ["filipino", "philippine"],
    "MY": ["malaysian"],
    "SG": ["singaporean"],
    "KR": ["korean", "south korean"],
    "TW": ["taiwanese"],
    "PK": ["pakistani"],
    "AF": ["afghan"],
    "IR": ["iranian"],
    "IQ": ["iraqi"],
    "SY": ["syrian"],
    "EG": ["egyptian"],
    "MA": ["moroccan"],
    "TN": ["tunisian"],
    "DZ": ["algerian"],
    "ET": ["ethiopian"],
    "KE": ["kenyan"],
    "NG": ["nigerian"],
    "ZA": ["south african"],
    "UG": ["ugandan"],
    "TZ": ["tanzanian"],
    "ZM": ["zambian"],
    "ZW": ["zimbabwean"],
    "CO": ["colombian"],
    "PE": ["peruvian"],
    "CL": ["chilean"],
    "BO": ["bolivian"],
    "PY": ["paraguayan"],
    "UY": ["uruguayan"],
    "EC": ["ecuadorean", "ecuadorian"],
    "VE": ["venezuelan"],
    "SV": ["salvadoran"],
    "CU": ["cuban"],
    "JM": ["jamaican"],
    "NZ": ["new zealand", "new zealander"],
}


MANUAL_CITIES = {
    "FR": ["paris", "lyon", "marseille"],
    "GB": ["london", "manchester", "edinburgh", "glasgow"],
    "DE": ["berlin", "munich", "hamburg", "frankfurt"],
    "JP": ["tokyo", "osaka", "kyoto"],
    "CN": ["beijing", "shanghai", "guangzhou"],
    "ES": ["madrid", "barcelona", "seville", "valencia"],
    "PT": ["lisbon", "porto"],
    "IT": ["rome", "milan", "naples", "venice"],
    "RU": ["moscow", "saint petersburg", "st petersburg"],
    "UA": ["kyiv", "kiev"],
    "IL": ["tel aviv", "jerusalem"],
    "IN": ["delhi", "new delhi", "mumbai", "bangalore", "kolkata"],
    "TH": ["bangkok"],
    "VN": ["hanoi", "ho chi minh city", "saigon"],
    "ID": ["jakarta", "bali"],
    "PH": ["manila"],
    "MY": ["kuala lumpur"],
    "SG": ["singapore"],
    "KR": ["seoul", "busan"],
    "TW": ["taipei"],
    "AU": ["sydney", "melbourne", "brisbane"],
    "NZ": ["auckland", "wellington"],
    "CA": ["toronto", "vancouver", "montreal", "ottawa"],
    "US": ["new york", "los angeles", "san francisco", "chicago", "seattle", "washington dc"],
    "MX": ["mexico city"],
    "BR": ["sao paulo", "rio de janeiro"],
    "AR": ["buenos aires"],
    "CL": ["santiago"],
    "PE": ["lima"],
    "CO": ["bogota"],
    "VE": ["caracas"],
    "NL": ["amsterdam"],
    "BE": ["brussels"],
    "AT": ["vienna"],
    "CH": ["zurich", "geneva"],
    "SE": ["stockholm"],
    "NO": ["oslo"],
    "DK": ["copenhagen"],
    "FI": ["helsinki"],
    "IE": ["dublin"],
    "PL": ["warsaw"],
    "CZ": ["prague"],
    "HU": ["budapest"],
    "GR": ["athens"],
    "TR": ["istanbul", "ankara"],
    "EG": ["cairo"],
    "KE": ["nairobi"],
    "NG": ["lagos"],
    "ZA": ["johannesburg", "cape town"],
    "MA": ["casablanca", "marrakesh"],
    "AE": ["dubai", "abu dhabi"],
    "QA": ["doha"],
    "SA": ["riyadh"],
    "HK": ["hong kong"],
}


EXCLUDED_ALIAS_NORMS = {
    "jersey",
    "georgia",
    "jordan",
    "chad",
    "oman",
    "turkey",
    "and",
    "the",
}


MATCH_TYPE_TO_TIER = {
    "country": "high",
    "demonym": "medium",
    "city": "medium",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("-", " ")
    cleaned = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith(("P", "S")):
            cleaned.append(" ")
        else:
            cleaned.append(ch)
    return " ".join("".join(cleaned).split())


def strip_diacritics(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def add_entry(
    rows: list[dict[str, str]],
    alpha2: str,
    alpha3: str,
    canonical: str,
    alias: str,
    source: str,
    match_type: str,
) -> None:
    alias_norm = normalize(alias)
    if not alias_norm or alias_norm in EXCLUDED_ALIAS_NORMS or len(alias_norm) < 2:
        return
    rows.append(
        {
            "country_alpha2": alpha2,
            "country_alpha3": alpha3,
            "canonical_name": canonical,
            "alias": alias,
            "alias_norm": alias_norm,
            "source": source,
            "match_type": match_type,
            "confidence_tier": MATCH_TYPE_TO_TIER[match_type],
        }
    )


def add_name_variants(name: str) -> set[str]:
    variants = {name}
    folded = strip_diacritics(name)
    if folded and folded != name:
        variants.add(folded)
    return {variant for variant in variants if variant}


def is_safe_city_alias(alias: str, *, is_alternate: bool) -> bool:
    alias_norm = normalize(alias)
    if not alias_norm or alias_norm in EXCLUDED_ALIAS_NORMS:
        return False
    if re.search(r"\d", alias):
        return False
    ascii_only = all(ord(ch) < 128 for ch in alias_norm)
    if is_alternate and ascii_only and len(alias_norm) < 6 and " " not in alias_norm:
        return False
    if is_alternate and ascii_only and alias_norm.isalpha() and alias_norm in {
        "can",
        "she",
        "may",
        "sir",
        "ant",
        "san",
    }:
        return False
    return True


def add_babel_country_names(
    rows: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
) -> None:
    locales = [Locale.parse(code) for code in LOCALES]
    for country in pycountry.countries:
        alpha2 = getattr(country, "alpha_2", "")
        alpha3 = getattr(country, "alpha_3", "")
        canonical = getattr(country, "name", "")
        for locale in locales:
            translated = locale.territories.get(alpha2)
            if not translated:
                continue
            for alias in add_name_variants(str(translated)):
                key = (alpha2, normalize(alias), "country")
                if key not in seen:
                    add_entry(rows, alpha2, alpha3, canonical, alias, f"babel:{locale.language}", "country")
                    seen.add(key)


def add_geonames_city_aliases(
    rows: list[dict[str, str]],
    seen: set[tuple[str, str, str]],
) -> None:
    gc = geonamescache.GeonamesCache()
    country_info = gc.get_countries()
    candidate_aliases: dict[str, set[tuple[str, str, str]]] = defaultdict(set)

    for code, info in country_info.items():
        capital = info.get("capital")
        if not capital:
            continue
        for alias in add_name_variants(str(capital)):
            if is_safe_city_alias(alias, is_alternate=False):
                alias_norm = normalize(alias)
                candidate_aliases[alias_norm].add((code, alias, "capital"))

    for alias_norm, entries in candidate_aliases.items():
        countries = {country for country, _, _ in entries}
        if len(countries) != 1:
            continue
        country = next(iter(countries))
        country_obj = pycountry.countries.get(alpha_2=country)
        if not country_obj:
            continue
        alpha3 = country_obj.alpha_3
        canonical = getattr(country_obj, "name", country)
        for _, alias, source_suffix in sorted(entries):
            key = (country, normalize(alias), "city")
            if key not in seen:
                add_entry(rows, country, alpha3, canonical, alias, f"geonames:{source_suffix}", "city")
                seen.add(key)


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    for country in pycountry.countries:
        alpha2 = getattr(country, "alpha_2", "")
        alpha3 = getattr(country, "alpha_3", "")
        canonical = getattr(country, "name", "")
        names = {
            canonical,
            getattr(country, "official_name", ""),
            getattr(country, "common_name", ""),
        }
        for name in names:
            if not name:
                continue
            for alias in add_name_variants(name):
                key = (alpha2, normalize(alias), "country")
                if key not in seen:
                    add_entry(rows, alpha2, alpha3, canonical, alias, "pycountry", "country")
                    seen.add(key)

    add_babel_country_names(rows, seen)

    for alpha2, aliases in MANUAL_ALIASES.items():
        country = pycountry.countries.get(alpha_2=alpha2)
        if not country:
            continue
        alpha3 = country.alpha_3
        canonical = getattr(country, "name", alpha2)
        for alias in aliases:
            for variant in add_name_variants(alias):
                key = (alpha2, normalize(variant), "country")
                if key not in seen:
                    add_entry(rows, alpha2, alpha3, canonical, variant, "manual_alias", "country")
                    seen.add(key)

    for alpha2, demonyms in MANUAL_DEMONYMS.items():
        country = pycountry.countries.get(alpha_2=alpha2)
        if not country:
            continue
        alpha3 = country.alpha_3
        canonical = getattr(country, "name", alpha2)
        for alias in demonyms:
            for variant in add_name_variants(alias):
                key = (alpha2, normalize(variant), "demonym")
                if key not in seen:
                    add_entry(rows, alpha2, alpha3, canonical, variant, "manual_demonym", "demonym")
                    seen.add(key)

    for alpha2, cities in MANUAL_CITIES.items():
        country = pycountry.countries.get(alpha_2=alpha2)
        if not country:
            continue
        alpha3 = country.alpha_3
        canonical = getattr(country, "name", alpha2)
        for alias in cities:
            for variant in add_name_variants(alias):
                key = (alpha2, normalize(variant), "city")
                if key not in seen:
                    add_entry(rows, alpha2, alpha3, canonical, variant, "manual_city", "city")
                    seen.add(key)

    add_geonames_city_aliases(rows, seen)

    rows.sort(key=lambda r: (r["country_alpha2"], r["match_type"], r["alias_norm"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a country lexicon from installed source packages.")
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "country_alpha2",
                "country_alpha3",
                "canonical_name",
                "alias",
                "alias_norm",
                "source",
                "match_type",
                "confidence_tier",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} lexicon rows to {args.output_csv}")


if __name__ == "__main__":
    main()
