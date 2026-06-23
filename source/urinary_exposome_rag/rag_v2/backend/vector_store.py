from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

from .config import CHROMA_COLLECTION, EMBEDDING_MODEL, VECTOR_DIR
from .data_loader import load_all_documents, split_semicolon
from .schemas import RagFilters, Source


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chroma_client() -> PersistentClient:
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_DIR))


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    clean: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def _limit_documents_balanced(docs: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    if not limit or limit <= 0 or len(docs) <= limit:
        return docs
    groups: dict[str, list[dict[str, Any]]] = {}
    for doc in docs:
        collection = str(doc.get("metadata", {}).get("collection", "unknown"))
        groups.setdefault(collection, []).append(doc)

    per_group = max(1, limit // max(1, len(groups)))
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for group_docs in groups.values():
        for doc in group_docs[:per_group]:
            selected.append(doc)
            selected_ids.add(str(doc["id"]))

    for doc in docs:
        if len(selected) >= limit:
            break
        if str(doc["id"]) not in selected_ids:
            selected.append(doc)
            selected_ids.add(str(doc["id"]))

    return selected[:limit]


def build_index(batch_size: int = 64, reset: bool = False, limit: int | None = None) -> dict[str, Any]:
    client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
    collection = get_collection()
    docs = load_all_documents()
    docs = _limit_documents_balanced(docs, limit)
    if not docs:
        raise RuntimeError("No documents were found. Build the RAG corpus first.")

    model = get_embedding_model()
    total = len(docs)
    for start in range(0, total, batch_size):
        batch = docs[start : start + batch_size]
        texts = [doc["document"] for doc in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
        collection.upsert(
            ids=[doc["id"] for doc in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[_clean_metadata(doc["metadata"]) for doc in batch],
        )
        print(f"Indexed {min(start + batch_size, total)}/{total} documents", flush=True)
    return {
        "collection": CHROMA_COLLECTION,
        "count": collection.count(),
        "indexed_this_run": len(docs),
        "vector_dir": str(VECTOR_DIR),
    }


def index_status() -> dict[str, Any]:
    collection = get_collection()
    return {
        "collection": CHROMA_COLLECTION,
        "count": collection.count(),
        "vector_dir": str(VECTOR_DIR),
        "embedding_model": EMBEDDING_MODEL,
    }


def _where_from_filters(filters: RagFilters) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if filters.source == "local_data":
        clauses.append({"source_group": "local_data"})
    elif filters.source == "literature":
        clauses.append({"source_group": "literature"})
    elif filters.source == "effects":
        clauses.append({"source_type": "effect_estimate"})
    elif filters.source == "fulltext":
        clauses.append({"collection": {"$in": ["fulltext_chunks", "fulltext_effects"]}})
    elif filters.source == "abstract":
        clauses.append({"collection": {"$in": ["abstract_chunks", "abstract_effects"]}})

    if filters.effects_only:
        clauses.append({"source_type": "effect_estimate"})
    if filters.table_only:
        clauses.append({"fulltext_location": {"$in": ["table", "xml_table"]}})
    if filters.chinese_only:
        clauses.append({"china_or_chinese_population_flag": "yes"})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _passes_semicolon_filter(metadata: dict[str, Any], filters: RagFilters) -> bool:
    if filters.exposure_domain != "all":
        if filters.exposure_domain not in split_semicolon(str(metadata.get("exposure_domains", ""))):
            return False
    if filters.disease_group != "all":
        if filters.disease_group not in split_semicolon(str(metadata.get("disease_groups", ""))):
            return False
    return True


def _effect_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    if metadata.get("source_type") != "effect_estimate":
        return {}
    keys = [
        "measure",
        "estimate",
        "ci_low",
        "ci_high",
        "p_operator",
        "p_value",
        "fdr",
        "exposure_candidates",
        "source_dataset",
        "population",
        "outcome",
        "disease",
        "category",
        "icd10",
    ]
    return {key: str(metadata.get(key, "")) for key in keys}


def _metadata_to_source(rank: int, score: float, doc_id: str, document: str, metadata: dict[str, Any]) -> Source:
    source_label = str(metadata.get("source_label", ""))
    if not source_label:
        if metadata.get("source_group") == "local_data":
            source_label = "Local UrologicalExpomics"
        else:
            pmid = metadata.get("pmid") or "no PMID"
            pmcid = metadata.get("pmcid")
            location = metadata.get("fulltext_location") or metadata.get("collection") or ""
            source_label = f"PMID {pmid} / {pmcid} / {location}" if pmcid else f"PMID {pmid} / {location}"

    return Source(
        rank=rank,
        score=round(score, 4),
        id=doc_id,
        title=str(metadata.get("title", "")),
        text=str(metadata.get("text", document)),
        source_group=str(metadata.get("source_group", "literature")),  # type: ignore[arg-type]
        source_group_label=str(metadata.get("source_group_label", "")),
        collection=str(metadata.get("collection", "")),
        source_type=str(metadata.get("source_type", "")),
        source_label=source_label,
        source_url=str(metadata.get("source_url", "")),
        pmid=str(metadata.get("pmid", "")),
        pmcid=str(metadata.get("pmcid", "")),
        doi=str(metadata.get("doi", "")),
        exposure_domains=split_semicolon(str(metadata.get("exposure_domains", ""))),
        disease_groups=split_semicolon(str(metadata.get("disease_groups", ""))),
        effect=_effect_from_metadata(metadata),
    )


def search(query: str, top_k: int, filters: RagFilters) -> list[Source]:
    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([query], normalize_embeddings=True, show_progress_bar=False).tolist()[0]
    fetch_k = min(max(top_k * 6, 30), 200)
    raw = collection.query(
        query_embeddings=[query_embedding],
        n_results=fetch_k,
        where=_where_from_filters(filters),
        include=["documents", "metadatas", "distances"],
    )

    ids = raw.get("ids", [[]])[0]
    docs = raw.get("documents", [[]])[0]
    metadatas = raw.get("metadatas", [[]])[0]
    distances = raw.get("distances", [[]])[0]

    sources: list[Source] = []
    for doc_id, document, metadata, distance in zip(ids, docs, metadatas, distances):
        metadata = metadata or {}
        if not _passes_semicolon_filter(metadata, filters):
            continue
        score = 1.0 - float(distance) if not math.isnan(float(distance)) else 0.0
        sources.append(_metadata_to_source(len(sources) + 1, score, str(doc_id), str(document), metadata))
        if len(sources) >= top_k:
            break

    return _balance_source_groups(sources, top_k, filters)


def _balance_source_groups(sources: list[Source], top_k: int, filters: RagFilters) -> list[Source]:
    if filters.source not in {"all", "effects"}:
        return sources[:top_k]
    local = [source for source in sources if source.source_group == "local_data"]
    literature = [source for source in sources if source.source_group == "literature"]
    if not local or not literature:
        return sources[:top_k]

    per_group = max(1, top_k // 2)
    selected = local[:per_group] + literature[:per_group]
    selected_ids = {source.id for source in selected}
    for source in sources:
        if len(selected) >= top_k:
            break
        if source.id not in selected_ids:
            selected.append(source)
            selected_ids.add(source.id)
    selected.sort(key=lambda item: item.score, reverse=True)
    for rank, source in enumerate(selected[:top_k], start=1):
        source.rank = rank
    return selected[:top_k]
