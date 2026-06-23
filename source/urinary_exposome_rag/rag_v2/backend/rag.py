from __future__ import annotations

import asyncio
import time

from .config import MAX_CONTEXT_CHARS
from .graph_store import search_graph
from .lightrag_bridge import query_lightrag
from .llm import build_prompt, call_vllm_chat
from .schemas import ChatRequest, ChatResponse, RagFilters, SearchResponse, Source
from .vector_store import search


def source_group_counts(sources: list[Source]) -> dict[str, int]:
    return {
        "local_data": sum(1 for source in sources if source.source_group == "local_data"),
        "literature": sum(1 for source in sources if source.source_group == "literature"),
    }


def format_effect(source: Source) -> str:
    effect = source.effect or {}
    if not effect:
        return f"- {source.title or source.source_label}: {source.text[:350]} [Source {source.rank}]"
    ci = ""
    if effect.get("ci_low") and effect.get("ci_high"):
        ci = f" (95% CI {effect.get('ci_low')}-{effect.get('ci_high')})"
    p_value = f", P{effect.get('p_operator', '')}{effect.get('p_value')}" if effect.get("p_value") else ""
    fdr = f", FDR={effect.get('fdr')}" if effect.get("fdr") else ""
    exposure = effect.get("exposure_candidates") or "未注明暴露"
    disease = effect.get("disease") or ", ".join(source.disease_groups) or "未注明疾病"
    return (
        f"- {exposure} -> {disease}: "
        f"{effect.get('measure', '')}={effect.get('estimate', '')}{ci}{p_value}{fdr}. "
        f"[Source {source.rank}]"
    )


def retrieval_answer(query: str, sources: list[Source]) -> str:
    if not sources:
        return "当前 Chroma/bge-m3 知识库中没有检索到相关证据。请尝试放宽筛选条件，或先重建向量索引。"

    local = [source for source in sources if source.source_group == "local_data"]
    literature = [source for source in sources if source.source_group == "literature"]
    lines = [
        "基于 Chroma + bge-m3 的检索结果，下面将本地数据与文献证据分开呈现。",
        "",
        "一、我的本地 UrologicalExpomics 数据",
    ]
    if local:
        lines.extend(format_effect(source) for source in local[:6])
    else:
        lines.append("- 当前结果中没有匹配的本地 UrologicalExpomics 证据。")

    lines.extend(["", "二、已发表论文／全文表格证据"])
    if literature:
        lines.extend(format_effect(source) for source in literature[:6])
    else:
        lines.append("- 当前结果中没有匹配的论文或全文表格证据。")

    lines.extend(
        [
            "",
            "三、注意事项",
            "- HR/OR/RR、置信区间、P 值和 FDR 来自结构化表格或机器抽取结果；正式写作前请回到原始论文或数据表核对。",
            "- 开启 vLLM/Qwen 且服务可用时，系统会基于同一批证据生成更自然的中文解释。",
        ]
    )
    return "\n".join(lines)


def run_search(query: str, top_k: int, filters: RagFilters) -> SearchResponse:
    started = time.time()
    sources = search(query=query, top_k=top_k, filters=filters)
    return SearchResponse(
        sources=sources,
        source_groups=source_group_counts(sources),
        elapsed_ms=int((time.time() - started) * 1000),
    )


async def run_chat(request: ChatRequest) -> ChatResponse:
    started = time.time()
    sources = search(query=request.query, top_k=request.top_k, filters=request.filters)
    fallback = retrieval_answer(request.query, sources)
    answer = fallback
    llm_used = False
    llm_error: str | None = None
    graph_paths: list[dict] = []
    graph_error: str | None = None
    lightrag_context: str | None = None
    lightrag_error: str | None = None

    if request.use_graph:
        try:
            graph_paths = await asyncio.to_thread(search_graph, request.query, request.top_k, request.filters)
        except Exception as error:
            graph_error = str(error)

    if request.use_lightrag:
        try:
            lightrag_context = await query_lightrag(request.query)
        except Exception as error:
            lightrag_error = str(error)

    if request.use_llm and sources:
        try:
            prompt = build_prompt(
                request.query,
                sources,
                MAX_CONTEXT_CHARS,
                graph_paths=graph_paths,
                lightrag_context=lightrag_context,
            )
            answer = await asyncio.to_thread(call_vllm_chat, prompt, request.temperature)
            llm_used = True
        except Exception as error:
            llm_error = f"vLLM/Qwen 不可用，已返回检索兜底答案：{error}"

    return ChatResponse(
        answer=answer,
        retrieval_answer=fallback if llm_used else None,
        sources=sources,
        source_groups=source_group_counts(sources),
        llm_used=llm_used,
        llm_error=llm_error,
        graph_paths=graph_paths,
        graph_error=graph_error,
        lightrag_context=lightrag_context,
        lightrag_error=lightrag_error,
        elapsed_ms=int((time.time() - started) * 1000),
    )
