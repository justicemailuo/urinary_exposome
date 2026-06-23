import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def stable_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:16]


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def split_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            sentence_break = max(text.rfind(". ", start, end), text.rfind("; ", start, end))
            if sentence_break > start + int(max_chars * 0.55):
                end = sentence_break + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def article_metadata(article: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_type": "pmc_fulltext",
        "source_url": article.get("source_url", ""),
        "pmid": article.get("pmid", ""),
        "pmcid": article.get("pmcid", ""),
        "doi": article.get("doi", ""),
        "title": article.get("title", ""),
        "exposure_domains": article.get("exposure_domains", []),
        "disease_groups": article.get("disease_groups", []),
    }


def build_chunks(structured_dir: Path, max_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for path in sorted(structured_dir.glob("*.json")):
        article = load_json(path)
        meta = article_metadata(article)
        pmid = meta["pmid"]
        pmcid = meta["pmcid"]

        abstract = article.get("abstract", "")
        if abstract:
            for index, text in enumerate(split_text(f"Title: {meta['title']}\nSection: Abstract\n{abstract}", max_chars, overlap_chars)):
                chunks.append(
                    {
                        "chunk_id": stable_id(pmid, pmcid, "abstract", str(index), text[:80]),
                        "text": text,
                        "fulltext_location": "abstract",
                        "section_title": "Abstract",
                        "table_id": "",
                        **meta,
                    }
                )

        for section_index, section in enumerate(article.get("sections", []), start=1):
            title = section.get("title", "") or f"Section {section_index}"
            source_text = f"Title: {meta['title']}\nSection: {title}\n{section.get('text', '')}"
            for index, text in enumerate(split_text(source_text, max_chars, overlap_chars)):
                chunks.append(
                    {
                        "chunk_id": stable_id(pmid, pmcid, "section", str(section_index), str(index), text[:80]),
                        "text": text,
                        "fulltext_location": "section",
                        "section_title": title,
                        "table_id": "",
                        **meta,
                    }
                )

        for table_index, table in enumerate(article.get("tables", []), start=1):
            table_id = table.get("table_id") or f"{pmid}_{pmcid}_xml_t{table_index}"
            label = table.get("label", "")
            caption = table.get("caption", "")
            source_text = (
                f"Title: {meta['title']}\n"
                f"Table: {label}\n"
                f"Caption: {caption}\n"
                f"Rows: {table.get('text', '')}"
            )
            for index, text in enumerate(split_text(source_text, max_chars, overlap_chars)):
                chunks.append(
                    {
                        "chunk_id": stable_id(pmid, pmcid, "table", table_id, str(index), text[:80]),
                        "text": text,
                        "fulltext_location": "table",
                        "section_title": "",
                        "table_id": table_id,
                        "table_label": label,
                        "table_caption": caption,
                        "table_csv_path": table.get("csv_path", ""),
                        **meta,
                    }
                )
    return chunks


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_markdown(path: Path, chunks: list[dict[str, Any]]) -> None:
    lines = ["# PMC Full-Text RAG Corpus", ""]
    for chunk in chunks:
        label = chunk.get("table_label") or chunk.get("section_title") or chunk.get("fulltext_location")
        lines.extend(
            [
                f"## {chunk.get('title', 'Untitled')}",
                "",
                f"- Chunk ID: `{chunk['chunk_id']}`",
                f"- PMID/PMCID: {chunk.get('pmid', '')} / {chunk.get('pmcid', '')}",
                f"- Location: {chunk.get('fulltext_location', '')} {label}",
                f"- Source: {chunk.get('source_url', '')}",
                "",
                chunk["text"],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG chunks from structured PMC full text and tables.")
    parser.add_argument("--structured-dir", default="data/fulltext/structured", help="Structured full-text JSON directory.")
    parser.add_argument("--output-dir", default="data/rag", help="Output RAG directory.")
    parser.add_argument("--max-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=200)
    args = parser.parse_args()

    chunks = build_chunks(Path(args.structured_dir), args.max_chars, args.overlap_chars)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "rag_fulltext_chunks.jsonl", chunks)
    write_markdown(output_dir / "rag_fulltext_documents.md", chunks)
    print(f"fulltext_chunks: {len(chunks)}")
    print(f"table_chunks: {sum(chunk['fulltext_location'] == 'table' for chunk in chunks)}")


if __name__ == "__main__":
    main()
