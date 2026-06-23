import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any


NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"

MEASURE_PATTERN = re.compile(
    r"\b(?P<measure>HR|hazard ratio|OR|odds ratio|RR|relative risk|risk ratio|IRR|incidence rate ratio)\b"
    r"(?:\s*\[[A-Z]+\])?"
    r"(?:\s*(?:was|were|of|=|,|:))*\s*"
    r"(?P<estimate>" + NUMBER + r")"
    r"(?:[,\s]*[\[\(]?(?:95\s*%\s*)?(?:CI|confidence interval)[,\s:]*"
    r"(?P<ci_low>" + NUMBER + r")\s*(?:-|to|,)\s*(?P<ci_high>" + NUMBER + r")\]?\)?)?"
    r"(?:[;,\s]+P\s*(?P<p_operator><|>|=|<=|>=|≤|≥)\s*(?P<p_value>" + NUMBER + r"))?",
    re.IGNORECASE,
)

ALT_CI_PATTERN = re.compile(
    r"\b(?P<measure>HR|OR|RR|IRR)\b\s*[,\s]*"
    r"(?P<estimate>" + NUMBER + r")\s*"
    r"[\[\(](?P<ci_low>" + NUMBER + r")\s*(?:-|to|,)\s*(?P<ci_high>" + NUMBER + r")",
    re.IGNORECASE,
)

P_PATTERN = re.compile(r"\bP\s*(?P<p_operator><|>|=|<=|>=|≤|≥)\s*(?P<p_value>" + NUMBER + r")", re.IGNORECASE)

CHINA_PATTERN = re.compile(
    r"\b(China|Chinese|Han Chinese|Taiwan|Hong Kong|Macau|Shanghai|Beijing|Guangzhou|Wuhan|Chengdu|Shenzhen|CHARLS)\b",
    re.IGNORECASE,
)

EXPOSURE_TERMS = {
    "lifestyle": [
        "smoking",
        "tobacco",
        "secondhand smoke",
        "alcohol",
        "diet",
        "dietary",
        "physical activity",
        "exercise",
        "sedentary",
        "sleep",
        "obesity",
        "body mass index",
        "BMI",
        "GLP-1",
    ],
    "environmental_pollution": [
        "air pollution",
        "PM2.5",
        "PM10",
        "particulate matter",
        "nitrogen dioxide",
        "NO2",
        "ozone",
        "O3",
        "arsenic",
        "cadmium",
        "lead",
        "mercury",
        "PFAS",
        "pesticide",
        "phthalate",
        "bisphenol",
        "water pollution",
        "drinking water",
    ],
    "climate_built_environment": [
        "temperature",
        "heat",
        "heatwave",
        "heat wave",
        "humidity",
        "climate",
        "green space",
        "greenness",
        "NDVI",
        "noise",
        "built environment",
        "urbanization",
        "traffic",
    ],
    "baseline_disease": [
        "hypertension",
        "diabetes",
        "diabetes mellitus",
        "cardiovascular",
        "metabolic syndrome",
        "gout",
        "hyperuricemia",
        "dyslipidemia",
        "comorbidity",
        "multimorbidity",
    ],
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_measure(measure: str) -> str:
    measure = measure.lower()
    if measure in {"hazard ratio", "hr"}:
        return "HR"
    if measure in {"odds ratio", "or"}:
        return "OR"
    if measure in {"relative risk", "risk ratio", "rr"}:
        return "RR"
    if measure in {"incidence rate ratio", "irr"}:
        return "IRR"
    return measure.upper()


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"[^.!?]+[.!?]?", text):
        sentence = match.group(0).strip()
        if sentence:
            spans.append((match.start(), match.end(), sentence))
    return spans


def normalize_numeric_text(text: str) -> str:
    return (
        text.replace("\u00b7", ".")
        .replace("\u2027", ".")
        .replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
    )


def snippet_for(text: str, start: int, end: int, width: int = 280) -> str:
    left = max(0, start - width // 2)
    right = min(len(text), end + width // 2)
    snippet = text[left:right].replace("\n", " ")
    return " ".join(snippet.split())


def nearby_exposure_terms(text: str, start: int, end: int) -> list[str]:
    window = text[max(0, start - 450) : min(len(text), end + 450)].lower()
    found: list[str] = []
    for terms in EXPOSURE_TERMS.values():
        for term in terms:
            if term.lower() in window and term not in found:
                found.append(term)
    return found


def nearby_p_value(text: str, start: int, end: int) -> tuple[str, str]:
    window = text[end : min(len(text), end + 120)]
    match = P_PATTERN.search(window)
    if not match:
        return "", ""
    return match.group("p_operator"), match.group("p_value")


def extract_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    text = normalize_numeric_text(" ".join([record.get("title", ""), record.get("abstract", "")]).strip())
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()

    for pattern in [MEASURE_PATTERN, ALT_CI_PATTERN]:
        for match in pattern.finditer(text):
            key = (match.start(), match.end(), match.group(0))
            if key in seen:
                continue
            seen.add(key)
            p_operator = match.groupdict().get("p_operator") or ""
            p_value = match.groupdict().get("p_value") or ""
            if not p_value:
                p_operator, p_value = nearby_p_value(text, match.start(), match.end())

            rows.append(
                {
                    "pmid": record.get("pmid", ""),
                    "title": record.get("title", ""),
                    "journal": record.get("journal", ""),
                    "publication_year": record.get("publication_year", ""),
                    "source_url": record.get("source_url", ""),
                    "doi": record.get("doi", ""),
                    "measure": normalize_measure(match.group("measure")),
                    "estimate": match.group("estimate"),
                    "ci_low": match.groupdict().get("ci_low") or "",
                    "ci_high": match.groupdict().get("ci_high") or "",
                    "p_operator": p_operator,
                    "p_value": p_value,
                    "exposure_domains": "; ".join(record.get("exposure_domains", [])),
                    "disease_groups": "; ".join(record.get("disease_groups", [])),
                    "specific_exposure_candidates": "; ".join(nearby_exposure_terms(text, match.start(), match.end())),
                    "china_or_chinese_population_flag": "yes" if CHINA_PATTERN.search(text) else "no",
                    "snippet": snippet_for(text, match.start(), match.end()),
                    "extraction_level": "abstract_machine_extracted",
                    "needs_manual_check": "yes",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pmid",
        "title",
        "journal",
        "publication_year",
        "measure",
        "estimate",
        "ci_low",
        "ci_high",
        "p_operator",
        "p_value",
        "exposure_domains",
        "disease_groups",
        "specific_exposure_candidates",
        "china_or_chinese_population_flag",
        "source_url",
        "doi",
        "snippet",
        "extraction_level",
        "needs_manual_check",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Extract candidate HR/OR/RR values from PubMed titles and abstracts.")
    parser.add_argument("--input", default="data/raw/pubmed_records.json", help="Input PubMed records JSON.")
    parser.add_argument("--output-dir", default="data/effects", help="Output directory for effect tables.")
    args = parser.parse_args()

    records = load_records(Path(args.input))
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.extend(extract_from_record(record))

    rows.sort(key=lambda row: (row["china_or_chinese_population_flag"] != "yes", row["publication_year"], row["pmid"]))
    high_confidence_rows = [
        row for row in rows if (row["ci_low"] and row["ci_high"]) or row["p_value"]
    ]
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "effect_estimates_abstract_extracted.csv", rows)
    write_json(output_dir / "effect_estimates_abstract_extracted.json", rows)
    write_csv(output_dir / "effect_estimates_high_confidence.csv", high_confidence_rows)
    write_json(output_dir / "effect_estimates_high_confidence.json", high_confidence_rows)

    print(f"records: {len(records)}")
    print(f"candidate_effect_estimates: {len(rows)}")
    print(f"high_confidence_effect_estimates: {len(high_confidence_rows)}")
    print(f"china_or_chinese_population_candidates: {sum(row['china_or_chinese_population_flag'] == 'yes' for row in rows)}")
    print(f"CSV: {output_dir / 'effect_estimates_abstract_extracted.csv'}")
    print(f"JSON: {output_dir / 'effect_estimates_abstract_extracted.json'}")


if __name__ == "__main__":
    main()
