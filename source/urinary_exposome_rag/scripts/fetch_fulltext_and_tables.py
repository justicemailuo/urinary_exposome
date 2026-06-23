import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import pdfplumber
import requests
from bs4 import BeautifulSoup


IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
EUROPE_PMC_XML_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def clean_text(text: str) -> str:
    return " ".join(text.split())


def request_with_retries(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> requests.Response:
    if url.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
        url = "https://ftp.ncbi.nlm.nih.gov/" + url.removeprefix("ftp://ftp.ncbi.nlm.nih.gov/")
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(1.5 * (attempt + 1))
                continue
            response.raise_for_status()
            return response
        except Exception as error:  # noqa: BLE001
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def batch(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def map_pmids_to_pmcids(pmids: list[str], email: str | None = None) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}
    for group in batch(pmids, 200):
        params: dict[str, Any] = {
            "ids": ",".join(group),
            "format": "json",
        }
        if email:
            params["email"] = email
        data = request_with_retries(IDCONV_URL, params=params).json()
        for record in data.get("records", []):
            pmid = str(record.get("pmid", ""))
            if pmid:
                mapping[pmid] = {
                    "pmcid": record.get("pmcid", ""),
                    "doi": record.get("doi", ""),
                    "status": record.get("status", ""),
                }
        time.sleep(0.35)
    return mapping


def fetch_fulltext_xml(pmcid: str) -> str | None:
    url = EUROPE_PMC_XML_URL.format(pmcid=pmcid)
    try:
        response = request_with_retries(url, timeout=35)
    except RuntimeError:
        return None
    text = response.text.strip()
    if not text or "not found" in text[:300].lower():
        return None
    return text


def fetch_oa_links(pmcid: str) -> dict[str, Any]:
    try:
        response = request_with_retries(PMC_OA_URL, params={"id": pmcid}, timeout=60)
    except RuntimeError as error:
        return {"pmcid": pmcid, "error": str(error), "links": []}

    soup = BeautifulSoup(response.text, "xml")
    links: list[dict[str, str]] = []
    for link in soup.find_all("link"):
        href = link.get("href", "")
        if not href:
            continue
        links.append(
            {
                "format": link.get("format", ""),
                "updated": link.get("updated", ""),
                "href": href,
            }
        )
    license_node = soup.find("license")
    return {
        "pmcid": pmcid,
        "license": license_node.text.strip() if license_node and license_node.text else "",
        "links": links,
    }


def choose_pdf_link(oa_links: dict[str, Any]) -> str:
    for link in oa_links.get("links", []):
        href = link.get("href", "")
        if link.get("format", "").lower() == "pdf" or href.lower().endswith(".pdf"):
            return href
    return ""


def fetch_pmc_article_pdf_link(pmcid: str) -> str:
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    try:
        response = request_with_retries(article_url, timeout=60)
    except RuntimeError:
        return ""
    soup = BeautifulSoup(response.text, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href", "")
        if ".pdf" in href.lower() or href.lower().endswith("/pdf/"):
            return urljoin(article_url, href)
    return ""


def download_pdf(url: str, output_path: Path) -> bool:
    if not url:
        return False
    try:
        response = request_with_retries(url, timeout=120)
    except RuntimeError:
        return False
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not response.content[:5] == b"%PDF-":
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return True


def tag_name(node: ET.Element) -> str:
    return node.tag.split("}", 1)[-1]


def element_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return clean_text(" ".join(node.itertext()))


def find_all_by_name(root: ET.Element, name: str) -> list[ET.Element]:
    return [node for node in root.iter() if tag_name(node) == name]


def find_first_by_name(root: ET.Element, name: str) -> ET.Element | None:
    for node in root.iter():
        if tag_name(node) == name:
            return node
    return None


def parse_table(table_wrap: ET.Element) -> dict[str, Any]:
    label = element_text(next((child for child in table_wrap if tag_name(child) == "label"), None))
    caption_node = next((child for child in table_wrap if tag_name(child) == "caption"), None)
    caption = element_text(caption_node)

    table_node = find_first_by_name(table_wrap, "table")
    rows: list[list[str]] = []
    if table_node is not None:
        for tr in find_all_by_name(table_node, "tr"):
            cells: list[str] = []
            for child in tr:
                if tag_name(child) in {"td", "th"}:
                    cells.append(element_text(child))
            if cells:
                rows.append(cells)

    table_text = " | ".join(["; ".join(row) for row in rows])
    footnotes = [element_text(fn) for fn in find_all_by_name(table_wrap, "fn")]
    return {
        "label": label,
        "caption": caption,
        "rows": rows,
        "footnotes": footnotes,
        "text": clean_text(" ".join([label, caption, table_text, " ".join(footnotes)])),
    }


def parse_fulltext_xml(xml_text: str) -> dict[str, Any]:
    root = ET.fromstring(xml_text.encode("utf-8"))
    article_title = element_text(find_first_by_name(root, "article-title"))
    abstract = element_text(find_first_by_name(root, "abstract"))

    sections: list[dict[str, str]] = []
    body = find_first_by_name(root, "body")
    if body is not None:
        for section in find_all_by_name(body, "sec"):
            title = element_text(next((child for child in section if tag_name(child) == "title"), None))
            paragraphs = [element_text(child) for child in section if tag_name(child) == "p"]
            text = clean_text(" ".join(paragraph for paragraph in paragraphs if paragraph))
            if text:
                sections.append({"title": title, "text": text})

    tables = [parse_table(table_wrap) for table_wrap in find_all_by_name(root, "table-wrap")]
    return {
        "title": article_title,
        "abstract": abstract,
        "sections": sections,
        "tables": tables,
        "body_text": clean_text(" ".join([abstract] + [section["text"] for section in sections])),
    }


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def write_table_csv(path: Path, table: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = table.get("rows", [])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        if table.get("label") or table.get("caption"):
            writer.writerow([table.get("label", ""), table.get("caption", "")])
        writer.writerows(rows)
        if table.get("footnotes"):
            writer.writerow([])
            for footnote in table["footnotes"]:
                writer.writerow(["footnote", footnote])


def extract_pdf(pdf_path: Path, output_dir: Path, pmid: str, pmcid: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pdf_path": str(pdf_path),
        "pages": 0,
        "text_chars": 0,
        "tables": [],
        "error": "",
    }
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page_texts: list[str] = []
            for page_index, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text:
                    page_texts.append(f"[page {page_index}]\n{page_text}")
                for table_index, table in enumerate(page.extract_tables() or [], start=1):
                    if not table:
                        continue
                    table_rows = [[clean_text(cell or "") for cell in row] for row in table]
                    if not any(any(cell for cell in row) for row in table_rows):
                        continue
                    table_id = f"{pmid}_{pmcid}_pdf_p{page_index}_t{table_index}"
                    table_csv = output_dir / "pdf_tables" / f"{safe_filename(table_id)}.csv"
                    table_csv.parent.mkdir(parents=True, exist_ok=True)
                    with table_csv.open("w", encoding="utf-8-sig", newline="") as handle:
                        csv.writer(handle).writerows(table_rows)
                    result["tables"].append(
                        {
                            "table_id": table_id,
                            "page": page_index,
                            "csv_path": str(table_csv),
                            "rows": table_rows,
                            "text": " | ".join("; ".join(row) for row in table_rows),
                        }
                    )
            text = "\n\n".join(page_texts)
            text_path = output_dir / "pdf_text" / f"{safe_filename(pmid + '_' + pmcid)}.txt"
            text_path.parent.mkdir(parents=True, exist_ok=True)
            text_path.write_text(text, encoding="utf-8")
            result["pages"] = len(pdf.pages)
            result["text_chars"] = len(text)
            result["text_path"] = str(text_path)
    except Exception as error:  # noqa: BLE001
        result["error"] = str(error)
    return result


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pmid",
        "pmcid",
        "title",
        "fulltext_xml_available",
        "xml_table_count",
        "pdf_downloaded",
        "pdf_table_count",
        "license",
        "source_url",
        "xml_path",
        "pdf_path",
        "structured_path",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    configure_stdout()
    parser = argparse.ArgumentParser(description="Fetch open full text, PDFs, and tables for PubMed records.")
    parser.add_argument("--input", default="data/raw/pubmed_records.json", help="Input PubMed records JSON.")
    parser.add_argument("--output-dir", default="data/fulltext", help="Output full-text directory.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of records to try.")
    parser.add_argument("--pmids", default="", help="Optional comma-separated PMID allow-list.")
    parser.add_argument("--download-pdf", action="store_true", help="Download and parse OA PDFs when the OA service provides a PDF link.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    records = load_records(Path(args.input))
    if args.pmids:
        allow = {pmid.strip() for pmid in args.pmids.split(",") if pmid.strip()}
        records = [record for record in records if record.get("pmid") in allow]
    if args.limit:
        records = records[: args.limit]

    pmids = [record["pmid"] for record in records if record.get("pmid")]
    pmid_map = map_pmids_to_pmcids(pmids, os.environ.get("NCBI_EMAIL"))
    write_json(output_dir / "pmid_pmcid_map.json", pmid_map)

    manifest: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        pmid = record.get("pmid", "")
        pmcid = pmid_map.get(pmid, {}).get("pmcid", "")
        print(f"[{index}/{len(records)}] PMID {pmid} PMCID {pmcid or 'none'}", flush=True)
        if not pmcid:
            manifest.append(
                {
                    "pmid": pmid,
                    "pmcid": "",
                    "title": record.get("title", ""),
                    "fulltext_xml_available": "no",
                    "xml_table_count": 0,
                    "pdf_downloaded": "no",
                    "pdf_table_count": 0,
                    "license": "",
                    "source_url": record.get("source_url", ""),
                    "xml_path": "",
                    "pdf_path": "",
                    "structured_path": "",
                }
            )
            continue

        xml_text = ""
        structured: dict[str, Any] | None = None
        xml_path = ""
        structured_path = ""
        table_count = 0
        structured_path_obj = output_dir / "structured" / f"{safe_filename(pmid + '_' + pmcid)}.json"
        xml_path_obj = output_dir / "xml" / f"{safe_filename(pmid + '_' + pmcid)}.xml"
        if structured_path_obj.exists():
            structured = json.loads(structured_path_obj.read_text(encoding="utf-8"))
            structured_path = str(structured_path_obj)
            xml_path = str(xml_path_obj) if xml_path_obj.exists() else ""
            table_count = len(structured.get("tables", []))
            xml_text = xml_path_obj.read_text(encoding="utf-8") if xml_path_obj.exists() else ""
        else:
            xml_text = fetch_fulltext_xml(pmcid) or ""
        if not structured and xml_text:
            xml_path_obj.parent.mkdir(parents=True, exist_ok=True)
            xml_path_obj.write_text(xml_text, encoding="utf-8")
            xml_path = str(xml_path_obj)
            structured = parse_fulltext_xml(xml_text)
            structured.update(
                {
                    "pmid": pmid,
                    "pmcid": pmcid,
                    "source_url": record.get("source_url", ""),
                    "doi": record.get("doi", ""),
                    "exposure_domains": record.get("exposure_domains", []),
                    "disease_groups": record.get("disease_groups", []),
                }
            )
            write_json(structured_path_obj, structured)
            structured_path = str(structured_path_obj)

            for table_index, table in enumerate(structured.get("tables", []), start=1):
                table_id = f"{pmid}_{pmcid}_xml_t{table_index}"
                table["table_id"] = table_id
                table_csv = output_dir / "xml_tables" / f"{safe_filename(table_id)}.csv"
                write_table_csv(table_csv, table)
                table["csv_path"] = str(table_csv)
            write_json(structured_path_obj, structured)
            table_count = len(structured.get("tables", []))

        oa_links = fetch_oa_links(pmcid)
        pdf_path = ""
        pdf_table_count = 0
        pdf_downloaded = "no"
        if args.download_pdf:
            pdf_url = choose_pdf_link(oa_links)
            if pdf_url:
                pdf_path_obj = output_dir / "pdf" / f"{safe_filename(pmid + '_' + pmcid)}.pdf"
                downloaded = download_pdf(pdf_url, pdf_path_obj)
                if not downloaded:
                    fallback_pdf_url = fetch_pmc_article_pdf_link(pmcid)
                    downloaded = download_pdf(fallback_pdf_url, pdf_path_obj) if fallback_pdf_url else False
                if downloaded:
                    pdf_path = str(pdf_path_obj)
                    pdf_downloaded = "yes"
                    pdf_result = extract_pdf(pdf_path_obj, output_dir, pmid, pmcid)
                    write_json(output_dir / "pdf_extracted" / f"{safe_filename(pmid + '_' + pmcid)}.json", pdf_result)
                    pdf_table_count = len(pdf_result.get("tables", []))

        manifest.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "title": record.get("title", ""),
                "fulltext_xml_available": "yes" if xml_text else "no",
                "xml_table_count": table_count,
                "pdf_downloaded": pdf_downloaded,
                "pdf_table_count": pdf_table_count,
                "license": oa_links.get("license", ""),
                "source_url": record.get("source_url", ""),
                "xml_path": xml_path,
                "pdf_path": pdf_path,
                "structured_path": structured_path,
            }
        )
        time.sleep(0.35)

    write_manifest_csv(output_dir / "fulltext_manifest.csv", manifest)
    write_json(output_dir / "fulltext_manifest.json", manifest)
    print(f"records_tried: {len(records)}")
    print(f"with_pmcid: {sum(bool(row['pmcid']) for row in manifest)}")
    print(f"with_xml: {sum(row['fulltext_xml_available'] == 'yes' for row in manifest)}")
    print(f"xml_tables: {sum(int(row['xml_table_count']) for row in manifest)}")
    print(f"pdf_downloaded: {sum(row['pdf_downloaded'] == 'yes' for row in manifest)}")
    print(f"pdf_tables: {sum(int(row['pdf_table_count']) for row in manifest)}")


if __name__ == "__main__":
    main()
