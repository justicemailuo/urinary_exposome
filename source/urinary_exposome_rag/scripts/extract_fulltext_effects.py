import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from extract_effect_estimates import (
    CHINA_PATTERN,
    EXPOSURE_TERMS,
    MEASURE_PATTERN,
    P_PATTERN,
    nearby_exposure_terms,
    normalize_measure,
    normalize_numeric_text,
)


NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+)"
ESTIMATE_CI_PATTERN = re.compile(
    r"(?P<estimate>" + NUMBER + r")\s*[\(\[]\s*"
    r"(?P<ci_low>" + NUMBER + r")\s*(?:-|to|,)\s*(?P<ci_high>" + NUMBER + r")\s*[\)\]]",
    re.IGNORECASE,
)
MEASURE_CONTEXT_PATTERN = re.compile(
    r"\b(HR|OR|RR|IRR)\b|\b(hazard ratio|odds ratio|relative risk|risk ratio|incidence rate ratio)\b",
)


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def p_value_near(text: str, end: int) -> tuple[str, str]:
    match = P_PATTERN.search(text[end : min(len(text), end + 160)])
    if not match:
        return "", ""
    return match.group("p_operator"), match.group("p_value")


def exposure_terms_in_text(text: str) -> list[str]:
    lower = text.lower()
    found: list[str] = []
    for terms in EXPOSURE_TERMS.values():
        for term in terms:
            if term.lower() in lower and term not in found:
                found.append(term)
    return found


def infer_measure(context: str) -> str:
    match = MEASURE_CONTEXT_PATTERN.search(context)
    if not match:
        return ""
    return normalize_measure(match.group(1) or match.group(2))


def is_valid_measure_token(token: str) -> bool:
    if token in {"HR", "OR", "RR", "IRR"}:
        return True
    return token.lower() in {"hazard ratio", "odds ratio", "relative risk", "risk ratio", "incidence rate ratio"}


def make_row(
    article: dict[str, Any],
    source_location: str,
    location_label: str,
    text: str,
    measure: str,
    estimate: str,
    ci_low: str,
    ci_high: str,
    p_operator: str,
    p_value: str,
) -> dict[str, Any]:
    return {
        "pmid": article.get("pmid", ""),
        "pmcid": article.get("pmcid", ""),
        "title": article.get("title", ""),
        "source_url": article.get("source_url", ""),
        "doi": article.get("doi", ""),
        "measure": measure,
        "estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_operator": p_operator,
        "p_value": p_value,
        "exposure_domains": "; ".join(article.get("exposure_domains", [])),
        "disease_groups": "; ".join(article.get("disease_groups", [])),
        "specific_exposure_candidates": "; ".join(exposure_terms_in_text(text)),
        "china_or_chinese_population_flag": "yes" if CHINA_PATTERN.search(text) else "no",
        "source_location": source_location,
        "location_label": location_label,
        "snippet": clean_text(text[:1200]),
        "extraction_level": "fulltext_table_or_section_machine_extracted",
        "needs_manual_check": "yes",
    }


def extract_effects_from_text(article: dict[str, Any], source_location: str, location_label: str, text: str) -> list[dict[str, Any]]:
    text = normalize_numeric_text(clean_text(text))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for match in MEASURE_PATTERN.finditer(text):
        if not is_valid_measure_token(match.group("measure")):
            continue
        p_operator = match.groupdict().get("p_operator") or ""
        p_value = match.groupdict().get("p_value") or ""
        if not p_value:
            p_operator, p_value = p_value_near(text, match.end())
        key = (normalize_measure(match.group("measure")), match.group("estimate"), match.groupdict().get("ci_low") or "", match.groupdict().get("ci_high") or "")
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            make_row(
                article,
                source_location,
                location_label,
                text,
                normalize_measure(match.group("measure")),
                match.group("estimate"),
                match.groupdict().get("ci_low") or "",
                match.groupdict().get("ci_high") or "",
                p_operator,
                p_value,
            )
        )

    inferred_measure = infer_measure(text)
    if inferred_measure:
        for match in ESTIMATE_CI_PATTERN.finditer(text):
            key = (inferred_measure, match.group("estimate"), match.group("ci_low"), match.group("ci_high"))
            if key in seen:
                continue
            seen.add(key)
            p_operator, p_value = p_value_near(text, match.end())
            rows.append(
                make_row(
                    article,
                    source_location,
                    location_label,
                    text,
                    inferred_measure,
                    match.group("estimate"),
                    match.group("ci_low"),
                    match.group("ci_high"),
                    p_operator,
                    p_value,
                )
            )
    return rows


def extract_article(article_path: Path) -> list[dict[str, Any]]:
    article = load_json(article_path)
    rows: list[dict[str, Any]] = []
    abstract = article.get("abstract", "")
    if abstract:
        rows.extend(extract_effects_from_text(article, "xml_abstract", "abstract", abstract))
    for section in article.get("sections", []):
        label = section.get("title", "") or "section"
        rows.extend(extract_effects_from_text(article, "xml_section", label, section.get("text", "")))
    for table in article.get("tables", []):
        label = " ".join(part for part in [table.get("label", ""), table.get("caption", "")] if part)
        rows.extend(extract_effects_from_text(article, "xml_table", label, table.get("text", "")))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pmid",
        "pmcid",
        "title",
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
        "source_location",
        "location_label",
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
    configure_stdout()
    parser = argparse.ArgumentParser(description="Extract candidate HR/OR/RR values from full-text XML sections and tables.")
    parser.add_argument("--structured-dir", default="data/fulltext/structured", help="Directory with structured full-text JSON files.")
    parser.add_argument("--output-dir", default="data/effects", help="Output directory.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in sorted(Path(args.structured_dir).glob("*.json")):
        rows.extend(extract_article(path))

    rows.sort(key=lambda row: (row["source_location"] != "xml_table", row["pmid"], row["location_label"], row["measure"], row["estimate"]))
    high_confidence = [row for row in rows if row["ci_low"] and row["ci_high"]]

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "effect_estimates_fulltext_extracted.csv", rows)
    write_json(output_dir / "effect_estimates_fulltext_extracted.json", rows)
    write_csv(output_dir / "effect_estimates_fulltext_high_confidence.csv", high_confidence)
    write_json(output_dir / "effect_estimates_fulltext_high_confidence.json", high_confidence)

    print(f"structured_articles: {len(list(Path(args.structured_dir).glob('*.json')))}")
    print(f"candidate_effect_estimates: {len(rows)}")
    print(f"high_confidence_with_ci: {len(high_confidence)}")
    print(f"from_tables: {sum(row['source_location'] == 'xml_table' for row in rows)}")


if __name__ == "__main__":
    main()
