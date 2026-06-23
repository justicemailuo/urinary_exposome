import argparse
import json
import sys
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run a local TF-IDF retrieval smoke test over RAG chunks.")
    parser.add_argument("--chunks", default="data/rag/rag_chunks.jsonl", help="Path to RAG JSONL chunks.")
    parser.add_argument("--query", required=True, help="Question or retrieval query.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return.")
    args = parser.parse_args()

    chunks = load_jsonl(Path(args.chunks))
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=80000)
    matrix = vectorizer.fit_transform(texts)
    query_vector = vectorizer.transform([args.query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    ranked = scores.argsort()[::-1][: args.top_k]

    for rank, index in enumerate(ranked, start=1):
        chunk = chunks[int(index)]
        preview = " ".join(chunk["text"].split())[:700]
        print(f"\n[{rank}] score={scores[index]:.4f}")
        print(f"title: {chunk.get('title', '')}")
        print(f"source: {chunk.get('source_url') or chunk.get('source_type', '')}")
        print(f"exposure_domains: {', '.join(chunk.get('exposure_domains', []))}")
        print(f"disease_groups: {', '.join(chunk.get('disease_groups', []))}")
        print(f"text: {preview}")


if __name__ == "__main__":
    main()
