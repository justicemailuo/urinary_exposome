import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]


def normalize_space(text: str) -> str:
    return " ".join(text.split())


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = normalize_space(text)
    if len(text) <= max_chars:
        return [text] if text else []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_break = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if sentence_break > start + int(max_chars * 0.55):
                end = sentence_break + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def record_to_text(record: dict[str, Any]) -> str:
    authors = record.get("authors", [])
    first_author = authors[0] if authors else ""
    abstract = record.get("abstract", "")
    sections = [
        f"Title: {record.get('title', '')}",
        f"Source: PubMed PMID {record.get('pmid', '')}; {record.get('journal', '')}; {record.get('publication_year', '')}.",
        f"Authors: {first_author}{' et al.' if len(authors) > 1 else ''}",
        f"Exposure domains: {', '.join(record.get('exposure_domains', []))}",
        f"Disease groups: {', '.join(record.get('disease_groups', []))}",
        f"Publication types: {', '.join(record.get('publication_types', []))}",
        f"MeSH terms: {', '.join(record.get('mesh_terms', []))}",
        f"Keywords: {', '.join(record.get('keywords', []))}",
        f"Abstract: {abstract}",
    ]
    return "\n".join(section for section in sections if section.strip())


def build_chunks(records: list[dict[str, Any]], max_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for record in records:
        pmid = record.get("pmid", "")
        if not pmid:
            continue
        full_text = record_to_text(record)
        for index, text in enumerate(split_text(full_text, max_chars, overlap_chars)):
            chunks.append(
                {
                    "chunk_id": stable_id(pmid, str(index), text[:80]),
                    "text": text,
                    "source_type": "pubmed",
                    "source_url": record.get("source_url", ""),
                    "pmid": pmid,
                    "doi": record.get("doi", ""),
                    "title": record.get("title", ""),
                    "journal": record.get("journal", ""),
                    "publication_year": record.get("publication_year", ""),
                    "publication_date": record.get("publication_date", ""),
                    "exposure_domains": record.get("exposure_domains", []),
                    "disease_groups": record.get("disease_groups", []),
                    "query_labels": record.get("query_labels", []),
                    "publication_types": record.get("publication_types", []),
                    "mesh_terms": record.get("mesh_terms", []),
                    "keywords": record.get("keywords", []),
                    "chunk_index": index,
                }
            )
    return chunks


def add_taxonomy_chunks(chunks: list[dict[str, Any]]) -> None:
    taxonomy_notes = [
        {
            "title": "Urinary exposome RAG scope",
            "text": (
                "This knowledge base treats the exposome as external macro-level and individual-level exposures "
                "that may influence urinary system outcomes. Core exposure domains include lifestyle factors, "
                "environmental pollution, climate and built environment, and baseline diseases or comorbidities. "
                "Core outcomes include chronic kidney disease, acute kidney injury, urolithiasis, bladder cancer, "
                "renal cancer, urinary tract infection, and lower urinary tract symptoms."
            ),
        },
        {
            "title": "Mechanism map for urinary exposome studies",
            "text": (
                "Common mechanistic pathways linking exposures to urinary diseases include oxidative stress, "
                "systemic and renal inflammation, endothelial dysfunction, renal tubular injury, endocrine disruption, "
                "immune dysregulation, dehydration and urine concentration, DNA damage, and changes in uric acid, "
                "glucose, blood pressure, and lipid metabolism."
            ),
        },
        {
            "title": "Recommended epidemiologic modeling concepts",
            "text": (
                "For RAG retrieval and later analysis, keep exposure timing, spatial resolution, confounding, "
                "effect modification, and multi-exposure mixture methods explicit. Common methods include logistic "
                "regression, Cox models, generalized additive models, distributed lag nonlinear models, weighted "
                "quantile sum regression, Bayesian kernel machine regression, and causal inference sensitivity analyses."
            ),
        },
    ]
    for index, note in enumerate(taxonomy_notes):
        chunks.append(
            {
                "chunk_id": stable_id("taxonomy", str(index), note["title"]),
                "text": f"Title: {note['title']}\nSource: Project curated taxonomy.\n{note['text']}",
                "source_type": "curated_taxonomy",
                "source_url": "",
                "pmid": "",
                "doi": "",
                "title": note["title"],
                "journal": "",
                "publication_year": "",
                "publication_date": "",
                "exposure_domains": ["lifestyle", "environmental_pollution", "climate_built_environment", "baseline_disease"],
                "disease_groups": [
                    "chronic_kidney_disease",
                    "acute_kidney_injury",
                    "urolithiasis",
                    "bladder_cancer",
                    "renal_cancer",
                    "urinary_tract_infection",
                    "lower_urinary_tract_symptoms",
                ],
                "query_labels": ["curated_taxonomy"],
                "publication_types": [],
                "mesh_terms": [],
                "keywords": [],
                "chunk_index": 0,
            }
        )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, chunks: list[dict[str, Any]]) -> None:
    lines = ["# Urinary Exposome RAG Corpus", ""]
    for chunk in chunks:
        lines.extend(
            [
                f"## {chunk.get('title', 'Untitled')}",
                "",
                f"- Chunk ID: `{chunk['chunk_id']}`",
                f"- Source: {chunk.get('source_url') or chunk.get('source_type')}",
                f"- Exposure domains: {', '.join(chunk.get('exposure_domains', []))}",
                f"- Disease groups: {', '.join(chunk.get('disease_groups', []))}",
                "",
                chunk["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG-ready JSONL chunks from PubMed records.")
    parser.add_argument("--input", default="data/raw/pubmed_records.json", help="Input PubMed records JSON.")
    parser.add_argument("--output-dir", default="data/rag", help="Output RAG directory.")
    parser.add_argument("--max-chars", type=int, default=1600, help="Maximum characters per chunk.")
    parser.add_argument("--overlap-chars", type=int, default=180, help="Character overlap between chunks.")
    args = parser.parse_args()

    records = load_json(Path(args.input))
    chunks = build_chunks(records, args.max_chars, args.overlap_chars)
    add_taxonomy_chunks(chunks)

    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "rag_chunks.jsonl", chunks)
    write_markdown(output_dir / "rag_documents.md", chunks)

    print(f"Saved {len(chunks)} RAG chunks.")
    print(f"JSONL: {output_dir / 'rag_chunks.jsonl'}")
    print(f"MD:    {output_dir / 'rag_documents.md'}")


if __name__ == "__main__":
    main()
