from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = PROJECT_DIR.parents[1]
SOURCE_DATA_DIR = PROJECT_DIR / "data"
RELEASE_DIR = REPO_DIR / "releases"
PACKAGE_NAME = "urological_exposome_public_data_v0.1"


DROP_TEXT_COLUMNS = {"abstract", "snippet", "text", "document"}


def copy_csv_without_long_text(source: Path, target: Path, drop_columns: set[str] = DROP_TEXT_COLUMNS) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if not reader.fieldnames:
            raise ValueError(f"No header found in {source}")
        fieldnames = [name for name in reader.fieldnames if name not in drop_columns]
        rows = [{name: row.get(name, "") for name in fieldnames} for row in reader]
    with target.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    package_dir = RELEASE_DIR / PACKAGE_NAME
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    counts: dict[str, int] = {}
    files = {
        "data/tables/pubmed_records_metadata.csv": SOURCE_DATA_DIR / "tables" / "pubmed_records.csv",
        "data/effects/effect_estimates_high_confidence.csv": SOURCE_DATA_DIR / "effects" / "effect_estimates_high_confidence.csv",
        "data/effects/effect_estimates_fulltext_high_confidence.csv": SOURCE_DATA_DIR / "effects" / "effect_estimates_fulltext_high_confidence.csv",
        "data/effects/effect_estimates_urological_expomics_local.csv": SOURCE_DATA_DIR / "effects" / "effect_estimates_urological_expomics_local.csv",
    }

    for rel_target, source in files.items():
        if source.exists():
            counts[rel_target] = copy_csv_without_long_text(source, package_dir / rel_target)

    search_plan = PROJECT_DIR / "config" / "search_plan.json"
    if search_plan.exists():
        target = package_dir / "config" / "search_plan.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(search_plan, target)

    manifest = {
        "package": PACKAGE_NAME,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Sanitized public data package for UrologicalExpomics RAG. Long text fields such as abstracts and snippets are removed.",
        "counts": counts,
        "excluded": [
            "raw PubMed abstracts",
            "full-text XML/article text",
            "RAG text chunks",
            "Chroma/Neo4j/LightRAG database directories",
            ".env files and credentials",
        ],
    }
    write_text(package_dir / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    write_text(
        package_dir / "README.md",
        """# UrologicalExpomics Public Data Package

This package contains sanitized structured data for the UrologicalExpomics RAG workflow.

Included files:

- `data/tables/pubmed_records_metadata.csv`: PubMed metadata with abstracts removed.
- `data/effects/effect_estimates_high_confidence.csv`: high-confidence abstract-level effect estimates with snippets removed.
- `data/effects/effect_estimates_fulltext_high_confidence.csv`: high-confidence full-text/table effect estimates with snippets removed.
- `data/effects/effect_estimates_urological_expomics_local.csv`: local UrologicalExpomics effect-estimate records with snippets removed.
- `config/search_plan.json`: PubMed search configuration.
- `manifest.json`: package metadata and row counts.

This package is intended for evidence navigation and RAG index rebuilding. Machine-extracted effect estimates should be checked against the original sources before formal scientific interpretation.
""",
    )

    zip_path = RELEASE_DIR / f"{PACKAGE_NAME}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with ZipFile(zip_path, "w", ZIP_DEFLATED) as archive:
        for path in package_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(package_dir.parent))

    print(json.dumps({"package_dir": str(package_dir), "zip": str(zip_path), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
