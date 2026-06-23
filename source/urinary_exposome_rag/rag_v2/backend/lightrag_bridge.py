from __future__ import annotations

import asyncio
import threading
from typing import Any

from .config import (
    EMBEDDING_MODEL,
    LIGHTRAG_DIR,
    LIGHTRAG_ENABLED,
    LIGHTRAG_MODE,
    LIGHTRAG_TOP_K,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL,
)
from .data_loader import load_all_documents


def _imports():
    try:
        from lightrag import LightRAG, QueryParam
        from lightrag.llm.openai import openai_complete_if_cache
        from lightrag.utils import wrap_embedding_func_with_attrs
        return LightRAG, QueryParam, openai_complete_if_cache, wrap_embedding_func_with_attrs
    except ImportError as error:
        raise RuntimeError("LightRAG is not installed; run pip install -r requirements.txt") from error


_MODEL = None
_MODEL_LOCK = threading.Lock()


def _embedding_model():
    global _MODEL
    from sentence_transformers import SentenceTransformer
    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def create_lightrag():
    if not LIGHTRAG_ENABLED:
        raise RuntimeError("LightRAG is disabled; set RAG_LIGHTRAG_ENABLED=true")
    LightRAG, _, openai_complete_if_cache, wrap_embedding = _imports()

    @wrap_embedding(
        embedding_dim=1024,
        max_token_size=8192,
        model_name=EMBEDDING_MODEL,
        supports_asymmetric=True,
    )
    async def embed(texts: list[str], **_: Any):
        return await asyncio.to_thread(
            lambda: _embedding_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )

    async def complete(
        prompt: str,
        system_prompt: str | None = None,
        history_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ):
        return await openai_complete_if_cache(
            model=VLLM_MODEL,
            prompt=prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
            **kwargs,
        )

    LIGHTRAG_DIR.mkdir(parents=True, exist_ok=True)
    return LightRAG(
        working_dir=str(LIGHTRAG_DIR),
        kv_storage="JsonKVStorage",
        vector_storage="NanoVectorDBStorage",
        graph_storage="Neo4JStorage",
        doc_status_storage="JsonDocStatusStorage",
        llm_model_func=complete,
        llm_model_name=VLLM_MODEL,
        embedding_func=embed,
        chunk_token_size=400,
        chunk_overlap_token_size=50,
        max_extract_input_tokens=3000,
        entity_extract_max_gleaning=0,
        summary_context_size=3000,
        summary_max_tokens=400,
        addon_params={"language": "Chinese"},
        auto_manage_storages_states=True,
    )


async def index_lightrag(batch_size: int = 25, limit: int | None = None) -> dict[str, Any]:
    docs = load_all_documents()
    if limit:
        docs = docs[:limit]
    await asyncio.to_thread(_embedding_model)
    rag = create_lightrag()
    await rag.initialize_storages()
    try:
        indexed = 0
        for start in range(0, len(docs), batch_size):
            batch = docs[start : start + batch_size]
            await rag.ainsert(
                [doc["document"] for doc in batch],
                ids=[str(doc["id"]) for doc in batch],
            )
            indexed += len(batch)
        return {"indexed": indexed, "working_dir": str(LIGHTRAG_DIR)}
    finally:
        await rag.finalize_storages()


async def query_lightrag(query: str) -> str:
    _, QueryParam, _, _ = _imports()
    await asyncio.to_thread(_embedding_model)
    rag = create_lightrag()
    await rag.initialize_storages()
    try:
        result = await rag.aquery(
            query,
            param=QueryParam(
                mode=LIGHTRAG_MODE,
                top_k=LIGHTRAG_TOP_K,
                chunk_top_k=max(5, LIGHTRAG_TOP_K // 2),
                only_need_context=True,
                enable_rerank=False,
            ),
        )
        return str(result or "")
    finally:
        await rag.finalize_storages()


def lightrag_status() -> dict[str, Any]:
    return {
        "enabled": LIGHTRAG_ENABLED,
        "working_dir": str(LIGHTRAG_DIR),
        "mode": LIGHTRAG_MODE,
        "storage_exists": LIGHTRAG_DIR.exists() and any(LIGHTRAG_DIR.iterdir()),
    }
