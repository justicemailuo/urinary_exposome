from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .config import EFFECT_DIR, RAG_DIR


def split_semicolon(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def is_local_collection(collection: str) -> bool:
    return collection == "urological_expomics_local"


def source_group(collection: str) -> str:
    return "local_data" if is_local_collection(collection) else "literature"


def source_group_label(collection: str) -> str:
    if is_local_collection(collection):
        return "My local UrologicalExpomics data"
    return "Published literature / full-text tables"


def _load_jsonl(path: Path, collection: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = str(row.get("chunk_id") or f"{collection}:{line_no}")
            text = str(row.get("text", ""))
            title = str(row.get("title", ""))
            docs.append(
                {
                    "id": doc_id,
                    "document": " ".join([title, text]).strip(),
                    "metadata": {
                        "title": title,
                        "text": text,
                        "collection": collection,
                        "source_group": source_group(collection),
                        "source_group_label": source_group_label(collection),
                        "source_type": str(row.get("source_type", "chunk")),
                        "source_url": str(row.get("source_url", "")),
                        "pmid": str(row.get("pmid", "")),
                        "pmcid": str(row.get("pmcid", "")),
                        "doi": str(row.get("doi", "")),
                        "fulltext_location": str(row.get("fulltext_location", "")),
                        "table_id": str(row.get("table_id", "")),
                        "exposure_domains": ";".join(row.get("exposure_domains", []) or []),
                        "disease_groups": ";".join(row.get("disease_groups", []) or []),
                    },
                }
            )
    return docs


def _effect_source_label(row: dict[str, str], collection: str) -> str:
    if is_local_collection(collection):
        dataset = row.get("source_dataset") or "local analysis"
        population = row.get("population") or "all"
        outcome = row.get("outcome") or row.get("disease") or "outcome"
        return f"{dataset} / {population} / {outcome}"
    pmid = row.get("pmid") or "no PMID"
    pmcid = row.get("pmcid")
    location = row.get("source_location") or collection
    if pmcid:
        return f"PMID {pmid} / {pmcid} / {location}"
    return f"PMID {pmid} / {location}"


def _load_effect_csv(path: Path, collection: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    docs: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            measure = row.get("measure", "")
            estimate = row.get("estimate", "")
            ci_low = row.get("ci_low", "")
            ci_high = row.get("ci_high", "")
            p_value = row.get("p_value", "")
            fdr = row.get("fdr", "")
            exposure = row.get("exposure") or row.get("specific_exposure_candidates", "")
            disease = row.get("disease", "")
            source_label = _effect_source_label(row, collection)
            text_parts = [
                f"Effect estimate. Title: {row.get('title', '')}.",
                f"Dataset: {row.get('source_dataset', '')}.",
                f"Population: {row.get('population', '')}.",
                f"Outcome: {row.get('outcome', '')}.",
                f"Disease: {disease}.",
                f"Exposure: {exposure}.",
                f"Category: {row.get('category', '')}.",
                f"Measure: {measure}.",
                f"Estimate: {estimate}.",
                f"95% CI {ci_low}-{ci_high}." if ci_low and ci_high else "",
                f"P {row.get('p_operator', '')}{p_value}." if p_value else "",
                f"FDR {fdr}." if fdr else "",
                f"ICD10: {row.get('icd10', '')}.",
                f"Source: {source_label}.",
                f"Snippet: {row.get('snippet', '')}",
            ]
            document = " ".join(part for part in text_parts if part)
            doc_id = f"{collection}:{row.get('pmid', '')}:{index}"
            docs.append(
                {
                    "id": doc_id,
                    "document": document,
                    "metadata": {
                        "title": row.get("title", "") or f"{collection} effect estimate",
                        "text": document,
                        "collection": collection,
                        "source_group": source_group(collection),
                        "source_group_label": source_group_label(collection),
                        "source_type": "effect_estimate",
                        "source_label": source_label,
                        "source_url": row.get("source_url", ""),
                        "pmid": row.get("pmid", ""),
                        "pmcid": row.get("pmcid", ""),
                        "doi": row.get("doi", ""),
                        "fulltext_location": row.get("source_location", ""),
                        "table_id": row.get("location_label", ""),
                        "exposure_domains": row.get("exposure_domains", ""),
                        "disease_groups": row.get("disease_groups", ""),
                        "china_or_chinese_population_flag": row.get("china_or_chinese_population_flag", ""),
                        "measure": measure,
                        "estimate": estimate,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "p_operator": row.get("p_operator", ""),
                        "p_value": p_value,
                        "fdr": fdr,
                        "exposure_candidates": exposure,
                        "source_dataset": row.get("source_dataset", ""),
                        "population": row.get("population", ""),
                        "outcome": row.get("outcome", ""),
                        "disease": disease,
                        "category": row.get("category", ""),
                        "icd10": row.get("icd10", ""),
                    },
                }
            )
    return docs


def load_all_documents() -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    docs.extend(_load_jsonl(RAG_DIR / "rag_chunks.jsonl", "abstract_chunks"))
    docs.extend(_load_jsonl(RAG_DIR / "rag_fulltext_chunks.jsonl", "fulltext_chunks"))
    docs.extend(_load_effect_csv(EFFECT_DIR / "effect_estimates_high_confidence.csv", "abstract_effects"))
    docs.extend(_load_effect_csv(EFFECT_DIR / "effect_estimates_fulltext_high_confidence.csv", "fulltext_effects"))
    docs.extend(_load_effect_csv(EFFECT_DIR / "effect_estimates_urological_expomics_local.csv", "urological_expomics_local"))
    return docs
