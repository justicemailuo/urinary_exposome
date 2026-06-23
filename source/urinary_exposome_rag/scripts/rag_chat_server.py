import argparse
import csv
import html
import json
import math
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import webbrowser
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CN_QUERY_EXPANSIONS = {
    "慢性肾病": "chronic kidney disease CKD renal function eGFR albuminuria",
    "慢性肾脏病": "chronic kidney disease CKD renal function eGFR albuminuria",
    "急性肾损伤": "acute kidney injury AKI",
    "肾损伤": "kidney injury renal injury",
    "肾功能": "renal function kidney function eGFR",
    "结石": "urolithiasis nephrolithiasis kidney stone urinary stone",
    "泌尿系结石": "urolithiasis nephrolithiasis kidney stone urinary stone",
    "膀胱癌": "bladder cancer urinary bladder neoplasms",
    "肾癌": "renal cancer kidney cancer renal cell carcinoma",
    "尿路感染": "urinary tract infection UTI pyelonephritis",
    "前列腺增生": "benign prostatic hyperplasia BPH lower urinary tract symptoms LUTS",
    "下尿路症状": "lower urinary tract symptoms LUTS nocturia",
    "空气污染": "air pollution PM2.5 PM10 particulate matter NO2 ozone",
    "污染": "pollution air pollution water pollution heavy metals PFAS pesticide",
    "重金属": "heavy metals arsenic cadmium lead mercury",
    "砷": "arsenic",
    "镉": "cadmium",
    "铅": "lead",
    "汞": "mercury",
    "农药": "pesticide pesticides",
    "塑化剂": "phthalate phthalates bisphenol BPA",
    "全氟": "PFAS PFOA PFOS perfluoroalkyl substances",
    "饮水": "drinking water water pollution",
    "高温": "temperature heat heatwave heat wave",
    "热浪": "heatwave heat wave temperature heat",
    "气候": "climate climate change temperature humidity",
    "绿地": "green space greenness NDVI",
    "噪声": "noise traffic built environment",
    "生活方式": "lifestyle smoking alcohol diet physical activity sleep obesity BMI",
    "吸烟": "smoking tobacco secondhand smoke",
    "饮酒": "alcohol",
    "饮食": "diet dietary pattern",
    "运动": "physical activity exercise sedentary",
    "睡眠": "sleep",
    "肥胖": "obesity body mass index BMI",
    "基线疾病": "baseline disease comorbidity multimorbidity hypertension diabetes cardiovascular disease",
    "共病": "comorbidity multimorbidity baseline disease",
    "高血压": "hypertension blood pressure",
    "糖尿病": "diabetes diabetes mellitus hyperglycemia",
    "心血管": "cardiovascular disease CVD",
    "痛风": "gout hyperuricemia uric acid",
    "血脂": "dyslipidemia lipid",
    "华人": "China Chinese Taiwan Hong Kong Han Chinese",
    "中国": "China Chinese CHARLS Taiwan Hong Kong",
    "风险比": "hazard ratio HR relative risk RR risk ratio",
    "比值比": "odds ratio OR",
    "效应量": "HR OR RR IRR estimate confidence interval",
    "置信区间": "confidence interval 95% CI",
    "表格": "table xml_table source_location",
    "全文": "fulltext PMC Europe PMC",
}


@dataclass
class SearchResult:
    score: float
    item: dict[str, Any]


@dataclass
class LlmConfig:
    provider: str = "none"
    model: str = ""
    base_url: str = ""
    timeout: int = 120
    temperature: float = 0.1
    max_context_chars: int = 14000


class RagIndex:
    def __init__(self, project_dir: Path) -> None:
        self.project_dir = project_dir
        self.items = self._load_items()
        if not self.items:
            raise RuntimeError("No RAG documents found. Run the corpus build scripts first.")
        self.texts = [item["search_text"] for item in self.items]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=180000,
            min_df=1,
        )
        self.matrix = self.vectorizer.fit_transform(self.texts)
        self.exposure_domains = sorted({value for item in self.items for value in item.get("exposure_domains", []) if value})
        self.disease_groups = sorted({value for item in self.items for value in item.get("disease_groups", []) if value})
        self.source_types = sorted({item.get("source_type", "") for item in self.items if item.get("source_type")})

    def _load_jsonl(self, path: Path, source_label: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row["collection"] = source_label
                row["search_text"] = " ".join(
                    [
                        row.get("title", ""),
                        row.get("text", ""),
                        " ".join(row.get("exposure_domains", [])),
                        " ".join(row.get("disease_groups", [])),
                        row.get("fulltext_location", ""),
                    ]
                )
                rows.append(row)
        return rows

    def _load_effect_rows(self, path: Path, source_label: str) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                measure = row.get("measure", "")
                estimate = row.get("estimate", "")
                ci_low = row.get("ci_low", "")
                ci_high = row.get("ci_high", "")
                p_value = row.get("p_value", "")
                fdr = row.get("fdr", "")
                source_dataset = row.get("source_dataset", "")
                population = row.get("population", "")
                outcome = row.get("outcome", "")
                disease = row.get("disease", "")
                exposure = row.get("exposure", "") or row.get("specific_exposure_candidates", "")
                category = row.get("category", "")
                ci_text = f"95% CI {ci_low}-{ci_high}" if ci_low and ci_high else ""
                p_text = f"P {row.get('p_operator', '')}{p_value}" if p_value else ""
                fdr_text = f"FDR {fdr}" if fdr else ""
                text = (
                    f"Effect estimate. Title: {row.get('title', '')}. "
                    f"Dataset: {source_dataset}. Population: {population}. "
                    f"Outcome: {outcome}. Disease: {disease}. "
                    f"Exposure: {exposure}. Category: {category}. "
                    f"Measure: {measure}. Estimate: {estimate}. {ci_text}. {p_text}. "
                    f"{fdr_text}. ICD10: {row.get('icd10', '')}. "
                    f"Exposure candidates: {row.get('specific_exposure_candidates', '')}. "
                    f"Disease groups: {row.get('disease_groups', '')}. "
                    f"Location: {row.get('source_location', '')} {row.get('location_label', '')}. "
                    f"Snippet: {row.get('snippet', '')}"
                )
                rows.append(
                    {
                        "chunk_id": f"effect:{row.get('pmid', '')}:{len(rows)}:{source_label}",
                        "text": text,
                        "source_type": "effect_estimate",
                        "collection": source_label,
                        "source_url": row.get("source_url", ""),
                        "pmid": row.get("pmid", ""),
                        "pmcid": row.get("pmcid", ""),
                        "doi": row.get("doi", ""),
                        "title": row.get("title", ""),
                        "exposure_domains": split_semicolon(row.get("exposure_domains", "")),
                        "disease_groups": split_semicolon(row.get("disease_groups", "")),
                        "fulltext_location": row.get("source_location", ""),
                        "table_id": row.get("location_label", "") if row.get("source_location") == "xml_table" else "",
                        "effect": {
                            "measure": measure,
                            "estimate": estimate,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                        "p_operator": row.get("p_operator", ""),
                        "p_value": p_value,
                        "fdr": fdr,
                        "exposure_candidates": row.get("specific_exposure_candidates", ""),
                        "source_location": row.get("source_location", ""),
                        "location_label": row.get("location_label", ""),
                        "china_or_chinese_population_flag": row.get("china_or_chinese_population_flag", ""),
                        "source_dataset": source_dataset,
                        "population": population,
                        "outcome": outcome,
                        "disease": disease,
                        "category": category,
                        "icd10": row.get("icd10", ""),
                    },
                    "search_text": text,
                }
                )
        return rows

    def _load_items(self) -> list[dict[str, Any]]:
        rag_dir = self.project_dir / "data" / "rag"
        effect_dir = self.project_dir / "data" / "effects"
        items: list[dict[str, Any]] = []
        items.extend(self._load_jsonl(rag_dir / "rag_chunks.jsonl", "abstract_chunks"))
        items.extend(self._load_jsonl(rag_dir / "rag_fulltext_chunks.jsonl", "fulltext_chunks"))
        items.extend(self._load_effect_rows(effect_dir / "effect_estimates_high_confidence.csv", "abstract_effects"))
        items.extend(self._load_effect_rows(effect_dir / "effect_estimates_fulltext_high_confidence.csv", "fulltext_effects"))
        items.extend(self._load_effect_rows(effect_dir / "effect_estimates_urological_expomics_local.csv", "urological_expomics_local"))
        return items

    def search(self, query: str, top_k: int, filters: dict[str, Any]) -> list[SearchResult]:
        expanded_query = expand_query(query)
        query_vector = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        candidates: list[SearchResult] = []
        for index, score in enumerate(scores):
            if score <= 0:
                continue
            item = self.items[index]
            if not passes_filters(item, filters):
                continue
            adjusted_score = float(score) + query_boost(query, expanded_query, item)
            candidates.append(SearchResult(score=adjusted_score, item=item))
        candidates.sort(key=lambda result: result.score, reverse=True)
        return balance_source_groups(candidates, top_k, filters)


def split_semicolon(value: str) -> list[str]:
    return [part.strip() for part in value.split(";") if part.strip()]


def balance_source_groups(candidates: list[SearchResult], top_k: int, filters: dict[str, Any]) -> list[SearchResult]:
    source = filters.get("source", "all")
    if source not in {"all", "effects"}:
        return candidates[:top_k]

    local = [result for result in candidates if is_local_item(result.item)]
    literature = [result for result in candidates if not is_local_item(result.item)]
    if not local or not literature:
        return candidates[:top_k]

    per_group = max(1, top_k // 2)
    selected = local[:per_group] + literature[:per_group]
    selected_ids = {id(result.item) for result in selected}

    for result in candidates:
        if len(selected) >= top_k:
            break
        if id(result.item) not in selected_ids:
            selected.append(result)
            selected_ids.add(id(result.item))

    selected.sort(key=lambda result: result.score, reverse=True)
    return selected[:top_k]


def expand_query(query: str) -> str:
    additions = []
    lower = query.lower()
    for cn, en in CN_QUERY_EXPANSIONS.items():
        if cn in query:
            additions.append(en)
    if "pm2.5" in lower or "pm 2.5" in lower:
        additions.append("PM2.5 fine particulate matter air pollution")
    if "or" in lower or "hr" in lower or "rr" in lower:
        additions.append("effect estimate odds ratio hazard ratio relative risk confidence interval")
    return " ".join([query] + additions)


def query_boost(original_query: str, expanded_query: str, item: dict[str, Any]) -> float:
    query_lower = expanded_query.lower()
    original_lower = original_query.lower()
    text_lower = item.get("search_text", "").lower()
    boost = 0.0

    exact_terms = ["pfas", "pm2.5", "pm10", "arsenic", "cadmium", "lead", "mercury", "pesticide", "phthalate", "bisphenol"]
    for term in exact_terms:
        if term in query_lower and term in text_lower:
            boost += 0.16

    disease_intents = {
        "renal_cancer": ["肾癌", "renal cancer", "kidney cancer", "renal cell carcinoma"],
        "bladder_cancer": ["膀胱癌", "bladder cancer"],
        "chronic_kidney_disease": ["慢性肾病", "慢性肾脏病", "ckd", "chronic kidney disease"],
        "acute_kidney_injury": ["急性肾损伤", "aki", "acute kidney injury"],
        "urolithiasis": ["结石", "kidney stone", "urolithiasis", "nephrolithiasis"],
        "urinary_tract_infection": ["尿路感染", "uti", "urinary tract infection"],
        "lower_urinary_tract_symptoms": ["下尿路症状", "前列腺增生", "luts", "bph"],
    }
    for disease, terms in disease_intents.items():
        if any(term.lower() in query_lower for term in terms) and disease in item.get("disease_groups", []):
            boost += 0.18

    if any(term in original_lower for term in ["hr", "or", "rr", "风险比", "比值比", "效应量", "置信区间"]):
        if item.get("source_type") == "effect_estimate":
            boost += 0.12
        if item.get("fulltext_location") in {"xml_table", "table"}:
            boost += 0.06

    if any(term in original_query for term in ["表格", "table"]) and item.get("fulltext_location") in {"xml_table", "table"}:
        boost += 0.12

    return boost


def passes_filters(item: dict[str, Any], filters: dict[str, Any]) -> bool:
    exposure = filters.get("exposure_domain", "all")
    disease = filters.get("disease_group", "all")
    source = filters.get("source", "all")
    table_only = bool(filters.get("table_only", False))
    effects_only = bool(filters.get("effects_only", False))
    chinese_only = bool(filters.get("chinese_only", False))

    if exposure != "all" and exposure not in item.get("exposure_domains", []):
        return False
    if disease != "all" and disease not in item.get("disease_groups", []):
        return False
    if source != "all":
        if source == "local_data" and not is_local_item(item):
            return False
        if source == "literature" and is_local_item(item):
            return False
        if source == "fulltext" and item.get("collection") not in {"fulltext_chunks", "fulltext_effects"}:
            return False
        if source == "abstract" and item.get("collection") not in {"abstract_chunks", "abstract_effects"}:
            return False
        if source == "effects" and item.get("source_type") != "effect_estimate":
            return False
    if table_only and item.get("fulltext_location") != "table" and item.get("fulltext_location") != "xml_table":
        return False
    if effects_only and item.get("source_type") != "effect_estimate":
        return False
    if chinese_only:
        effect = item.get("effect", {})
        text = " ".join([item.get("title", ""), item.get("text", ""), str(effect)])
        if effect.get("china_or_chinese_population_flag") != "yes" and not any(
            term in text.lower() for term in ["china", "chinese", "taiwan", "hong kong", "charls"]
        ):
            return False
    return True


def is_local_item(item: dict[str, Any]) -> bool:
    return item.get("collection") == "urological_expomics_local"


def evidence_group(item: dict[str, Any]) -> str:
    return "local_data" if is_local_item(item) else "literature"


def evidence_group_label(item: dict[str, Any]) -> str:
    return "My local UrologicalExpomics data" if is_local_item(item) else "Published literature / full-text tables"


def clip_text(text: str, max_chars: int = 850) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


def source_label(item: dict[str, Any]) -> str:
    effect = item.get("effect", {})
    if is_local_item(item):
        dataset = effect.get("source_dataset") or "local analysis"
        population = effect.get("population") or "all"
        outcome = effect.get("outcome") or effect.get("disease") or "outcome"
        return f"{dataset} / {population} / {outcome}"
    pmid = item.get("pmid") or "no PMID"
    pmcid = item.get("pmcid")
    location = item.get("fulltext_location") or item.get("collection") or item.get("source_type", "")
    if pmcid:
        return f"PMID {pmid} / {pmcid} / {location}"
    return f"PMID {pmid} / {location}"


def build_answer(query: str, results: list[SearchResult]) -> dict[str, Any]:
    if not results:
        return {
            "answer": "没有在当前知识库中检索到足够相关的证据。可以放宽筛选条件，或先扩大 PubMed/全文抓取数量。",
            "sources": [],
        }

    effect_results = [result for result in results if result.item.get("source_type") == "effect_estimate"]
    lines: list[str] = []
    lines.append("基于当前知识库检索结果，优先结论如下：")

    if effect_results:
        lines.append("")
        lines.append("效应量候选结果：")
        for result in effect_results[:6]:
            item = result.item
            effect = item.get("effect", {})
            ci = ""
            if effect.get("ci_low") and effect.get("ci_high"):
                ci = f"，95%CI {effect['ci_low']}-{effect['ci_high']}"
            p_value = ""
            if effect.get("p_value"):
                p_value = f"，P{effect.get('p_operator', '')}{effect['p_value']}"
            fdr_value = ""
            if effect.get("fdr"):
                fdr_value = f"，FDR={effect['fdr']}"
            exposure = effect.get("exposure_candidates") or "未在片段中明确识别"
            disease = ", ".join(item.get("disease_groups", [])) or "未标注"
            lines.append(
                f"- {exposure} -> {disease}：{effect.get('measure', '')}={effect.get('estimate', '')}{ci}{p_value}{fdr_value}。"
                f" 来源：{source_label(item)}。"
            )
    else:
        lines.append("")
        lines.append("没有在前排结果中找到结构化 HR/OR/RR 值，以下是最相关证据片段。")

    lines.append("")
    lines.append("相关证据片段：")
    for result in results[:5]:
        item = result.item
        title = item.get("title", "Untitled")
        text = clip_text(item.get("text", ""), 420)
        lines.append(f"- {title}：{text} 来源：{source_label(item)}。")

    lines.append("")
    lines.append("注意：这些是 RAG 检索和机器抽取结果，正式写作前需要回到原文/table 核对。")

    sources = []
    for rank, result in enumerate(results, start=1):
        item = result.item
        sources.append(
            {
                "rank": rank,
                "score": round(result.score, 4),
                "title": item.get("title", ""),
                "source_label": source_label(item),
                "source_url": item.get("source_url", ""),
                "pmid": item.get("pmid", ""),
                "pmcid": item.get("pmcid", ""),
                "collection": item.get("collection", ""),
                "source_type": item.get("source_type", ""),
                "fulltext_location": item.get("fulltext_location", ""),
                "exposure_domains": item.get("exposure_domains", []),
                "disease_groups": item.get("disease_groups", []),
                "snippet": clip_text(item.get("text", ""), 900),
                "effect": item.get("effect", {}),
            }
        )
    return {"answer": "\n".join(lines), "sources": sources}


def format_effect_line(item: dict[str, Any], include_source: bool) -> str:
    effect = item.get("effect", {})
    ci = ""
    if effect.get("ci_low") and effect.get("ci_high"):
        ci = f" (95% CI {effect['ci_low']}-{effect['ci_high']})"
    p_value = ""
    if effect.get("p_value"):
        p_value = f", P{effect.get('p_operator', '')}{effect['p_value']}"
    fdr_value = ""
    if effect.get("fdr"):
        fdr_value = f", FDR={effect['fdr']}"
    exposure = effect.get("exposure_candidates") or "unspecified exposure"
    disease = effect.get("disease") or ", ".join(item.get("disease_groups", [])) or "unspecified disease"
    measure = effect.get("measure", "")
    estimate = effect.get("estimate", "")
    source_text = f" Source: {source_label(item)}." if include_source else ""
    return f"- {exposure} -> {disease}: {measure}={estimate}{ci}{p_value}{fdr_value}.{source_text}"


def build_source_payload(results: list[SearchResult]) -> list[dict[str, Any]]:
    sources = []
    for rank, result in enumerate(results, start=1):
        item = result.item
        sources.append(
            {
                "rank": rank,
                "score": round(result.score, 4),
                "title": item.get("title", ""),
                "source_label": source_label(item),
                "source_url": item.get("source_url", ""),
                "pmid": item.get("pmid", ""),
                "pmcid": item.get("pmcid", ""),
                "collection": item.get("collection", ""),
                "source_group": evidence_group(item),
                "source_group_label": evidence_group_label(item),
                "source_type": item.get("source_type", ""),
                "fulltext_location": item.get("fulltext_location", ""),
                "exposure_domains": item.get("exposure_domains", []),
                "disease_groups": item.get("disease_groups", []),
                "snippet": clip_text(item.get("text", ""), 900),
                "effect": item.get("effect", {}),
            }
        )
    return sources


def build_grouped_answer(query: str, results: list[SearchResult]) -> dict[str, Any]:
    if not results:
        return {
            "answer": "没有在当前知识库中检索到足够相关的证据。可以放宽筛选条件，或扩大 PubMed/全文抓取数量。",
            "sources": [],
            "source_groups": {"local_data": 0, "literature": 0},
        }

    local_results = [result for result in results if is_local_item(result.item)]
    literature_results = [result for result in results if not is_local_item(result.item)]
    local_effects = [result for result in local_results if result.item.get("source_type") == "effect_estimate"]
    literature_effects = [result for result in literature_results if result.item.get("source_type") == "effect_estimate"]

    lines: list[str] = []
    lines.append("基于当前 RAG 检索结果，下面把“你的本地数据”和“论文/全文表格证据”分开讨论。")
    lines.append("")
    lines.append("一、你的本地 UrologicalExpomics 数据")
    if local_effects:
        for result in local_effects[:6]:
            lines.append(format_effect_line(result.item, include_source=False))
    else:
        lines.append("- 当前前排结果中没有检索到匹配的本地 HR/OR/RR 结构化结果。")

    lines.append("")
    lines.append("二、论文/全文表格证据")
    if literature_effects:
        for result in literature_effects[:6]:
            lines.append(format_effect_line(result.item, include_source=True))
    else:
        lines.append("- 当前前排结果中没有检索到匹配的论文 HR/OR/RR 结构化结果。")

    lines.append("")
    lines.append("三、相关证据片段")
    if local_results:
        lines.append("本地数据片段：")
        for result in local_results[:3]:
            item = result.item
            title = item.get("title", "Local UrologicalExpomics result")
            lines.append(f"- {title}: {clip_text(item.get('text', ''), 360)} Source: {source_label(item)}.")
    if literature_results:
        lines.append("论文证据片段：")
        for result in literature_results[:3]:
            item = result.item
            title = item.get("title", "Untitled")
            lines.append(f"- {title}: {clip_text(item.get('text', ''), 360)} Source: {source_label(item)}.")

    lines.append("")
    lines.append("注意：本地数据来自你的 UrologicalExpomics 分析结果；论文证据来自 PubMed/PMC 全文和表格抽取。机器抽取的 HR/OR/RR、CI、P 值、FDR 在正式写作前需要回到原始表格核对。")

    return {
        "answer": "\n".join(lines),
        "sources": build_source_payload(results),
        "source_groups": {
            "local_data": len(local_results),
            "literature": len(literature_results),
        },
    }


def build_llm_prompt(query: str, sources: list[dict[str, Any]], max_context_chars: int) -> str:
    context_blocks: list[str] = []
    used = 0
    for source in sources:
        effect = source.get("effect") or {}
        effect_text = ""
        if effect:
            ci = ""
            if effect.get("ci_low") and effect.get("ci_high"):
                ci = f", 95% CI {effect.get('ci_low')}-{effect.get('ci_high')}"
            p_value = ""
            if effect.get("p_value"):
                p_value = f", P{effect.get('p_operator', '')}{effect.get('p_value')}"
            fdr = f", FDR={effect.get('fdr')}" if effect.get("fdr") else ""
            effect_text = (
                f"Effect: {effect.get('exposure_candidates', '')} -> "
                f"{', '.join(source.get('disease_groups', []))}; "
                f"{effect.get('measure', '')}={effect.get('estimate', '')}{ci}{p_value}{fdr}."
            )
        block = (
            f"[Source {source.get('rank')}]\n"
            f"Evidence group: {source.get('source_group_label', '')}\n"
            f"Title: {source.get('title', '')}\n"
            f"Source label: {source.get('source_label', '')}\n"
            f"URL: {source.get('source_url', '')}\n"
            f"{effect_text}\n"
            f"Snippet: {source.get('snippet', '')}\n"
        )
        if used + len(block) > max_context_chars:
            break
        context_blocks.append(block)
        used += len(block)

    return (
        "You are a biomedical RAG assistant for urological exposomics. "
        "Answer in Chinese. Use only the provided evidence. "
        "Do not invent HR, OR, RR, confidence intervals, P values, FDR, PMID, or PMCID. "
        "When evidence is machine-extracted, explicitly say it needs manual verification. "
        "Discuss local UrologicalExpomics data and published literature/full-text table evidence in separate sections. "
        "Do not merge local results with paper results as if they came from the same source. "
        "Prioritize local UrologicalExpomics results when the question asks about UKB, NHANES, WeEndPd, or local data. "
        "Cite sources inline using [Source n].\n\n"
        f"Question:\n{query}\n\n"
        "Evidence:\n"
        + "\n".join(context_blocks)
        + "\n\nAnswer format:\n"
        "1. Local UrologicalExpomics data\n"
        "2. Published literature / full-text table evidence\n"
        "3. Comparison and interpretation\n"
        "4. Caveats and source list\n"
    )


def call_ollama(prompt: str, config: LlmConfig) -> str:
    base_url = (config.base_url or "http://127.0.0.1:11434").rstrip("/")
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": "You answer biomedical RAG questions in Chinese using only supplied evidence.",
            },
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {
            "temperature": config.temperature,
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/chat",
        data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=config.timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed.get("message", {}).get("content", "").strip()


def maybe_generate_with_llm(query: str, answer_payload: dict[str, Any], config: LlmConfig, use_llm: bool) -> dict[str, Any]:
    if not use_llm or config.provider == "none":
        return answer_payload
    if config.provider != "ollama":
        answer_payload["llm_error"] = f"Unsupported LLM provider: {config.provider}"
        return answer_payload
    if not config.model:
        answer_payload["llm_error"] = "No local LLM model configured."
        return answer_payload
    try:
        prompt = build_llm_prompt(query, answer_payload.get("sources", []), config.max_context_chars)
        llm_answer = call_ollama(prompt, config)
        if llm_answer:
            answer_payload["retrieval_answer"] = answer_payload.get("answer", "")
            answer_payload["answer"] = llm_answer
            answer_payload["llm_provider"] = config.provider
            answer_payload["llm_model"] = config.model
    except Exception as error:  # noqa: BLE001
        answer_payload["llm_error"] = f"LLM generation failed, returned retrieval answer instead: {error}"
    return answer_payload


def json_response(handler: BaseHTTPRequestHandler, payload: dict[str, Any], status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def html_response(handler: BaseHTTPRequestHandler, body: str) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    body = handler.rfile.read(length).decode("utf-8") if length else "{}"
    return json.loads(body or "{}")


def make_handler(index: RagIndex, llm_config: LlmConfig):
    class RagChatHandler(BaseHTTPRequestHandler):
        server_version = "UrinaryExposomeRagChat/1.0"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            sys.stdout.write("[%s] %s\n" % (self.log_date_time_string(), format % args))

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                html_response(self, APP_HTML)
                return
            if parsed.path == "/api/status":
                json_response(
                    self,
                    {
                        "items": len(index.items),
                        "exposure_domains": index.exposure_domains,
                        "disease_groups": index.disease_groups,
                        "source_types": index.source_types,
                        "llm_provider": llm_config.provider,
                        "llm_model": llm_config.model,
                    },
                )
                return
            json_response(self, {"error": "not found"}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path != "/api/chat":
                json_response(self, {"error": "not found"}, status=404)
                return
            try:
                payload = read_json_body(self)
                query = str(payload.get("query", "")).strip()
                if not query:
                    json_response(self, {"error": "query is required"}, status=400)
                    return
                filters = payload.get("filters", {}) if isinstance(payload.get("filters", {}), dict) else {}
                top_k = int(payload.get("top_k", 8))
                top_k = max(3, min(top_k, 20))
                started = time.time()
                results = index.search(query, top_k=top_k, filters=filters)
                answer = build_grouped_answer(query, results)
                answer = maybe_generate_with_llm(
                    query=query,
                    answer_payload=answer,
                    config=llm_config,
                    use_llm=bool(payload.get("use_llm", False)),
                )
                answer["elapsed_ms"] = int((time.time() - started) * 1000)
                answer["query"] = query
                answer["filters"] = filters
                json_response(self, answer)
            except Exception as error:  # noqa: BLE001
                json_response(
                    self,
                    {"error": str(error), "traceback": traceback.format_exc(limit=5)},
                    status=500,
                )

    return RagChatHandler


APP_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>泌尿暴露组 RAG</title>
  <style>
    :root {
      --bg: #f5f7f8;
      --panel: #ffffff;
      --ink: #1d252c;
      --muted: #60707d;
      --line: #d9e0e4;
      --accent: #166a5b;
      --accent-dark: #0f4c43;
      --warn: #8a5a12;
      --source: #eef4f2;
      --shadow: 0 10px 28px rgba(21, 32, 38, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      letter-spacing: 0;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
    }
    aside {
      background: #18242b;
      color: #edf4f2;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
      border-right: 1px solid #0f171c;
    }
    .brand {
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding-bottom: 14px;
      border-bottom: 1px solid rgba(255,255,255,0.14);
    }
    .brand h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.25;
      font-weight: 650;
    }
    .brand span {
      color: #b7c8c3;
      font-size: 13px;
    }
    label {
      display: block;
      font-size: 13px;
      color: #cbd9d5;
      margin-bottom: 7px;
    }
    select, input[type="range"] {
      width: 100%;
    }
    select {
      height: 38px;
      border: 1px solid rgba(255,255,255,0.16);
      border-radius: 6px;
      background: #24343d;
      color: #f7fbfa;
      padding: 0 10px;
      outline: none;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 14px;
      color: #edf4f2;
      margin: 8px 0;
    }
    .check input {
      width: 16px;
      height: 16px;
      accent-color: var(--accent);
    }
    .status {
      margin-top: auto;
      border-top: 1px solid rgba(255,255,255,0.14);
      padding-top: 14px;
      color: #b7c8c3;
      font-size: 13px;
      line-height: 1.5;
    }
    main {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: 100vh;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 16px 22px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    header h2 {
      margin: 0;
      font-size: 18px;
      font-weight: 650;
    }
    .meta {
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }
    .chat {
      overflow: auto;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .msg {
      max-width: 980px;
      border-radius: 8px;
      padding: 14px 15px;
      line-height: 1.6;
      white-space: pre-wrap;
      box-shadow: var(--shadow);
    }
    .user {
      align-self: flex-end;
      background: var(--accent);
      color: white;
      max-width: min(760px, 90%);
    }
    .assistant {
      align-self: flex-start;
      background: var(--panel);
      border: 1px solid var(--line);
      width: min(980px, 100%);
    }
    .sources {
      display: grid;
      gap: 10px;
      margin-top: 12px;
    }
    .source {
      border: 1px solid var(--line);
      background: var(--source);
      border-radius: 8px;
      padding: 10px;
    }
    .source-title {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 6px;
    }
    .source-title strong {
      font-size: 14px;
    }
    .score {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .source p {
      margin: 0;
      color: #32414a;
      font-size: 13px;
      line-height: 1.45;
    }
    .source a {
      color: var(--accent-dark);
      text-decoration: none;
      font-weight: 600;
    }
    .composer {
      background: var(--panel);
      border-top: 1px solid var(--line);
      padding: 14px 22px 18px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 10px;
    }
    textarea {
      resize: none;
      height: 74px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      outline: none;
      line-height: 1.45;
    }
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(22, 106, 91, 0.12);
    }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: white;
      padding: 0 18px;
      min-width: 94px;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled {
      background: #9cafaa;
      cursor: default;
    }
    .empty {
      margin: auto;
      color: var(--muted);
      text-align: center;
      max-width: 620px;
      line-height: 1.7;
    }
    @media (max-width: 860px) {
      .app { grid-template-columns: 1fr; }
      aside {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        padding: 14px;
      }
      .brand, .status { grid-column: 1 / -1; }
      main { height: calc(100vh - 280px); min-height: 640px; }
      header { align-items: flex-start; flex-direction: column; }
      .composer { grid-template-columns: 1fr; }
      button { height: 44px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="brand">
        <h1>泌尿暴露组 RAG</h1>
        <span>摘要、全文、table、效应量联合检索</span>
      </div>
      <div>
        <label for="exposure">暴露域</label>
        <select id="exposure"></select>
      </div>
      <div>
        <label for="disease">疾病结局</label>
        <select id="disease"></select>
      </div>
      <div>
        <label for="source">来源</label>
        <select id="source">
          <option value="all">全部</option>
          <option value="fulltext">全文 / table</option>
          <option value="abstract">摘要</option>
          <option value="effects">HR / OR / RR</option>
        </select>
      </div>
      <div>
        <label for="topk">召回数量 <span id="topkLabel">8</span></label>
        <input id="topk" type="range" min="3" max="20" value="8" />
      </div>
      <div>
        <label class="check"><input id="tableOnly" type="checkbox" />只看 table</label>
        <label class="check"><input id="effectsOnly" type="checkbox" />只看效应量</label>
        <label class="check"><input id="chineseOnly" type="checkbox" />中国 / 华人</label>
      </div>
      <div class="status" id="status">正在载入索引...</div>
    </aside>
    <main>
      <header>
        <h2>交互式证据问答</h2>
        <div class="meta" id="meta">未查询</div>
      </header>
      <section class="chat" id="chat">
        <div class="empty">可以直接问：PM2.5 和慢性肾病有什么关系？PFAS 与肾癌的 RR/OR 值有哪些？中国人群里高温和肾功能下降的证据是什么？</div>
      </section>
      <form class="composer" id="form">
        <textarea id="query" placeholder="输入问题，例如：帮我找 PFAS 和肾癌的 HR/OR/RR 以及 95%CI"></textarea>
        <button id="send" type="submit">发送</button>
      </form>
    </main>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const form = document.getElementById("form");
    const query = document.getElementById("query");
    const send = document.getElementById("send");
    const statusBox = document.getElementById("status");
    const meta = document.getElementById("meta");
    const topk = document.getElementById("topk");
    const topkLabel = document.getElementById("topkLabel");

    function optionList(select, values, label) {
      select.innerHTML = "";
      const all = document.createElement("option");
      all.value = "all";
      all.textContent = label;
      select.appendChild(all);
      values.forEach(value => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
    }

    function addMessage(role, text, sources) {
      const empty = chat.querySelector(".empty");
      if (empty) empty.remove();
      const msg = document.createElement("div");
      msg.className = `msg ${role}`;
      msg.textContent = text;
      if (sources && sources.length) {
        const list = document.createElement("div");
        list.className = "sources";
        sources.forEach(src => {
          const item = document.createElement("div");
          item.className = "source";
          const title = document.createElement("div");
          title.className = "source-title";
          const strong = document.createElement("strong");
          strong.textContent = `${src.rank}. ${src.title || src.source_label}`;
          const score = document.createElement("span");
          score.className = "score";
          score.textContent = `score ${src.score}`;
          title.appendChild(strong);
          title.appendChild(score);
          const p = document.createElement("p");
          const link = src.source_url ? `<a href="${src.source_url}" target="_blank" rel="noreferrer">${src.source_label}</a>` : src.source_label;
          p.innerHTML = `${link}<br>${escapeHtml(src.snippet || "")}`;
          item.appendChild(title);
          item.appendChild(p);
          list.appendChild(item);
        });
        msg.appendChild(list);
      }
      chat.appendChild(msg);
      chat.scrollTop = chat.scrollHeight;
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#039;"
      })[ch]);
    }

    function filters() {
      return {
        exposure_domain: document.getElementById("exposure").value,
        disease_group: document.getElementById("disease").value,
        source: document.getElementById("source").value,
        table_only: document.getElementById("tableOnly").checked,
        effects_only: document.getElementById("effectsOnly").checked,
        chinese_only: document.getElementById("chineseOnly").checked
      };
    }

    async function loadStatus() {
      const response = await fetch("/api/status");
      const data = await response.json();
      optionList(document.getElementById("exposure"), data.exposure_domains, "全部暴露");
      optionList(document.getElementById("disease"), data.disease_groups, "全部疾病");
      statusBox.textContent = `索引 ${data.items} 个片段\n暴露域 ${data.exposure_domains.length} 个\n疾病组 ${data.disease_groups.length} 个`;
    }

    topk.addEventListener("input", () => {
      topkLabel.textContent = topk.value;
    });

    query.addEventListener("keydown", event => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        form.requestSubmit();
      }
    });

    form.addEventListener("submit", async event => {
      event.preventDefault();
      const text = query.value.trim();
      if (!text) return;
      addMessage("user", text);
      query.value = "";
      send.disabled = true;
      send.textContent = "检索中";
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({query: text, filters: filters(), top_k: Number(topk.value)})
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "请求失败");
        }
        addMessage("assistant", data.answer, data.sources);
        meta.textContent = `${data.sources.length} 条来源 · ${data.elapsed_ms} ms`;
      } catch (error) {
        addMessage("assistant", `出错：${error.message}`);
        meta.textContent = "查询失败";
      } finally {
        send.disabled = false;
        send.textContent = "发送";
        query.focus();
      }
    });

    loadStatus().catch(error => {
      statusBox.textContent = `索引载入失败：${error.message}`;
    });
  </script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an interactive local RAG chat app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--project-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--open", action="store_true", help="Open the app in the default browser.")
    parser.add_argument("--llm-provider", default=os.environ.get("RAG_LLM_PROVIDER", "none"), choices=["none", "ollama"])
    parser.add_argument("--llm-model", default=os.environ.get("RAG_LLM_MODEL", ""))
    parser.add_argument("--llm-base-url", default=os.environ.get("RAG_LLM_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--llm-timeout", type=int, default=int(os.environ.get("RAG_LLM_TIMEOUT", "120")))
    parser.add_argument("--llm-temperature", type=float, default=float(os.environ.get("RAG_LLM_TEMPERATURE", "0.1")))
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    project_dir = Path(args.project_dir).resolve()
    print(f"Loading RAG index from {project_dir} ...", flush=True)
    index = RagIndex(project_dir)
    print(f"Loaded {len(index.items)} searchable items.", flush=True)
    llm_config = LlmConfig(
        provider=args.llm_provider,
        model=args.llm_model,
        base_url=args.llm_base_url,
        timeout=args.llm_timeout,
        temperature=args.llm_temperature,
    )
    if llm_config.provider != "none":
        print(
            f"LLM generation enabled via {llm_config.provider}: {llm_config.model} at {llm_config.base_url}",
            flush=True,
        )

    server = ThreadingHTTPServer((args.host, args.port), make_handler(index, llm_config))
    url = f"http://{args.host}:{args.port}"
    print(f"RAG chat server running at {url}", flush=True)
    if args.open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
