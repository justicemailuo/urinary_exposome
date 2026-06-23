import argparse
import csv
import json
import os
import time
import urllib.parse
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import requests


EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def request_json(url: str, params: dict[str, Any], sleep_seconds: float) -> dict[str, Any]:
    response = requests.get(url, params=params, timeout=40)
    response.raise_for_status()
    time.sleep(sleep_seconds)
    return response.json()


def request_text(url: str, params: dict[str, Any], sleep_seconds: float) -> str:
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    time.sleep(sleep_seconds)
    return response.text


def pubmed_search(
    term: str,
    retmax: int,
    email: str | None,
    api_key: str | None,
    sleep_seconds: float,
) -> tuple[list[str], int]:
    params: dict[str, Any] = {
        "db": "pubmed",
        "term": term,
        "retmode": "json",
        "retmax": retmax,
        "sort": "relevance",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    data = request_json(f"{EUTILS_BASE}/esearch.fcgi", params, sleep_seconds)
    result = data.get("esearchresult", {})
    return result.get("idlist", []), int(result.get("count", 0))


def pubmed_fetch(
    pmids: list[str],
    email: str | None,
    api_key: str | None,
    sleep_seconds: float,
) -> list[dict[str, Any]]:
    if not pmids:
        return []

    params: dict[str, Any] = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    xml_text = request_text(f"{EUTILS_BASE}/efetch.fcgi", params, sleep_seconds)
    root = ET.fromstring(xml_text)
    return [parse_pubmed_article(article) for article in root.findall(".//PubmedArticle")]


def text_or_empty(element: ET.Element | None) -> str:
    if element is None or element.text is None:
        return ""
    return " ".join(element.text.split())


def collect_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(" ".join(element.itertext()).split())


def parse_pub_date(pub_date: ET.Element | None) -> tuple[str, str]:
    if pub_date is None:
        return "", ""
    year = text_or_empty(pub_date.find("Year"))
    month = text_or_empty(pub_date.find("Month"))
    day = text_or_empty(pub_date.find("Day"))
    medline = text_or_empty(pub_date.find("MedlineDate"))
    if year:
        return year, "-".join(part for part in [year, month, day] if part)
    return "", medline


def parse_abstract(article: ET.Element) -> str:
    parts: list[str] = []
    for item in article.findall(".//Abstract/AbstractText"):
        label = item.attrib.get("Label")
        text = collect_text(item)
        if not text:
            continue
        parts.append(f"{label}: {text}" if label else text)
    return "\n".join(parts)


def parse_authors(article: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article.findall(".//AuthorList/Author"):
        collective = text_or_empty(author.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        last = text_or_empty(author.find("LastName"))
        fore = text_or_empty(author.find("ForeName"))
        name = " ".join(part for part in [fore, last] if part)
        if name:
            authors.append(name)
    return authors


def parse_pubmed_article(article: ET.Element) -> dict[str, Any]:
    medline = article.find("MedlineCitation")
    pmid = text_or_empty(medline.find("PMID") if medline is not None else None)
    article_node = medline.find("Article") if medline is not None else None

    title = collect_text(article_node.find("ArticleTitle") if article_node is not None else None)
    journal = collect_text(article_node.find("Journal/Title") if article_node is not None else None)
    pub_year, pub_date = parse_pub_date(article_node.find("Journal/JournalIssue/PubDate") if article_node is not None else None)
    abstract = parse_abstract(article)

    doi = ""
    for article_id in article.findall(".//ArticleIdList/ArticleId"):
        if article_id.attrib.get("IdType") == "doi":
            doi = collect_text(article_id)
            break

    publication_types = [
        collect_text(node)
        for node in article.findall(".//PublicationTypeList/PublicationType")
        if collect_text(node)
    ]
    mesh_terms = [
        collect_text(node.find("DescriptorName"))
        for node in article.findall(".//MeshHeadingList/MeshHeading")
        if collect_text(node.find("DescriptorName"))
    ]
    keywords = [
        collect_text(node)
        for node in article.findall(".//KeywordList/Keyword")
        if collect_text(node)
    ]

    return {
        "pmid": pmid,
        "title": title,
        "abstract": abstract,
        "journal": journal,
        "publication_year": pub_year,
        "publication_date": pub_date,
        "authors": parse_authors(article),
        "doi": doi,
        "publication_types": publication_types,
        "mesh_terms": mesh_terms,
        "keywords": keywords,
        "source_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
    }


def build_query(exposure_query: str, disease_query: str, start: str, end: str, language_filter: str) -> str:
    date_filter = f'("{start}"[Date - Publication] : "{end}"[Date - Publication])'
    return f"({exposure_query}) AND ({disease_query}) AND {date_filter} AND {language_filter}"


def merge_labels(record: dict[str, Any], query_meta: dict[str, str]) -> None:
    for field in ["exposure_domains", "disease_groups", "query_labels"]:
        record.setdefault(field, [])
    if query_meta["exposure_domain"] not in record["exposure_domains"]:
        record["exposure_domains"].append(query_meta["exposure_domain"])
    if query_meta["disease_group"] not in record["disease_groups"]:
        record["disease_groups"].append(query_meta["disease_group"])
    if query_meta["query_label"] not in record["query_labels"]:
        record["query_labels"].append(query_meta["query_label"])


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pmid",
        "title",
        "journal",
        "publication_year",
        "doi",
        "source_url",
        "exposure_domains",
        "disease_groups",
        "query_labels",
        "publication_types",
        "mesh_terms",
        "keywords",
        "abstract",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field, "") for field in fields}
            for field in ["exposure_domains", "disease_groups", "query_labels", "publication_types", "mesh_terms", "keywords"]:
                row[field] = "; ".join(record.get(field, []))
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PubMed records for urinary exposome RAG corpus.")
    parser.add_argument("--config", default="config/search_plan.json", help="Path to search plan JSON.")
    parser.add_argument("--output-dir", default="data", help="Output data directory.")
    parser.add_argument("--max-per-query", type=int, default=10, help="Maximum PubMed records to retrieve for each exposure x disease query.")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    config = load_json(config_path)
    email = os.environ.get("NCBI_EMAIL")
    api_key = os.environ.get("NCBI_API_KEY")
    sleep_seconds = 0.12 if api_key else 0.36

    start = config["date_range"]["start"]
    end = config["date_range"]["end"]
    language_filter = config.get("language_filter", "english[Language]")

    records_by_pmid: dict[str, dict[str, Any]] = {}
    search_log: list[dict[str, Any]] = []

    for exposure_key, exposure in config["exposure_domains"].items():
        for disease_key, disease in config["disease_groups"].items():
            query = build_query(exposure["query"], disease["query"], start, end, language_filter)
            query_label = f"{exposure_key}__{disease_key}"
            print(f"Searching {query_label} ...")
            pmids, hit_count = pubmed_search(query, args.max_per_query, email, api_key, sleep_seconds)
            fetched_records = pubmed_fetch(pmids, email, api_key, sleep_seconds)
            query_meta = {
                "exposure_domain": exposure_key,
                "disease_group": disease_key,
                "query_label": query_label,
            }

            for record in fetched_records:
                pmid = record.get("pmid")
                if not pmid:
                    continue
                if pmid not in records_by_pmid:
                    records_by_pmid[pmid] = record
                merge_labels(records_by_pmid[pmid], query_meta)

            search_log.append(
                {
                    "query_label": query_label,
                    "exposure_domain": exposure_key,
                    "exposure_label_zh": exposure.get("label_zh", ""),
                    "disease_group": disease_key,
                    "disease_label_zh": disease.get("label_zh", ""),
                    "query": query,
                    "encoded_query_url": "https://pubmed.ncbi.nlm.nih.gov/?" + urllib.parse.urlencode({"term": query}),
                    "hit_count": hit_count,
                    "retrieved_count": len(pmids),
                    "pmids": pmids,
                }
            )

    records = sorted(records_by_pmid.values(), key=lambda item: (item.get("publication_year", ""), item.get("pmid", "")), reverse=True)
    write_json(output_dir / "raw" / "pubmed_records.json", records)
    write_json(output_dir / "raw" / "search_log.json", search_log)
    write_csv(output_dir / "tables" / "pubmed_records.csv", records)

    print(f"Saved {len(records)} deduplicated PubMed records.")
    print(f"JSON: {output_dir / 'raw' / 'pubmed_records.json'}")
    print(f"CSV:  {output_dir / 'tables' / 'pubmed_records.csv'}")


if __name__ == "__main__":
    main()
