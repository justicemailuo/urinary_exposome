import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pmid",
        "pmcid",
        "title",
        "fulltext_xml_available",
        "xml_table_count",
        "xml_path",
        "structured_path",
        "source_url",
        "doi",
        "exposure_domains",
        "disease_groups",
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
    parser = argparse.ArgumentParser(description="Create a manifest from already downloaded full-text outputs.")
    parser.add_argument("--records", default="data/raw/pubmed_records.json")
    parser.add_argument("--fulltext-dir", default="data/fulltext")
    args = parser.parse_args()

    records = {record["pmid"]: record for record in load_json(Path(args.records))}
    fulltext_dir = Path(args.fulltext_dir)
    mapping_path = fulltext_dir / "pmid_pmcid_map.json"
    mapping = load_json(mapping_path) if mapping_path.exists() else {}

    rows: list[dict[str, Any]] = []
    for pmid, record in records.items():
        pmcid = mapping.get(pmid, {}).get("pmcid", "")
        structured_path = fulltext_dir / "structured" / f"{pmid}_{pmcid}.json" if pmcid else Path("")
        xml_path = fulltext_dir / "xml" / f"{pmid}_{pmcid}.xml" if pmcid else Path("")
        structured = load_json(structured_path) if pmcid and structured_path.exists() else {}
        rows.append(
            {
                "pmid": pmid,
                "pmcid": pmcid,
                "title": record.get("title", ""),
                "fulltext_xml_available": "yes" if structured else "no",
                "xml_table_count": len(structured.get("tables", [])) if structured else 0,
                "xml_path": str(xml_path) if xml_path.exists() else "",
                "structured_path": str(structured_path) if structured_path.exists() else "",
                "source_url": record.get("source_url", ""),
                "doi": record.get("doi", ""),
                "exposure_domains": "; ".join(record.get("exposure_domains", [])),
                "disease_groups": "; ".join(record.get("disease_groups", [])),
            }
        )

    write_csv(fulltext_dir / "fulltext_manifest_indexed.csv", rows)
    write_json(fulltext_dir / "fulltext_manifest_indexed.json", rows)
    print(f"records: {len(rows)}")
    print(f"with_pmcid: {sum(bool(row['pmcid']) for row in rows)}")
    print(f"with_fulltext_xml: {sum(row['fulltext_xml_available'] == 'yes' for row in rows)}")
    print(f"xml_tables: {sum(int(row['xml_table_count']) for row in rows)}")


if __name__ == "__main__":
    main()
