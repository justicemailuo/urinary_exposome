from __future__ import annotations

from typing import Any

import requests

from .config import LLM_TIMEOUT_SECONDS, VLLM_API_KEY, VLLM_BASE_URL, VLLM_MODEL
from .schemas import Source


def build_prompt(
    query: str,
    sources: list[Source],
    max_context_chars: int,
    graph_paths: list[dict[str, Any]] | None = None,
    lightrag_context: str | None = None,
) -> str:
    blocks: list[str] = []
    used = 0
    for source in sources:
        effect = source.effect or {}
        effect_text = ""
        if effect:
            ci = ""
            if effect.get("ci_low") and effect.get("ci_high"):
                ci = f", 95% CI {effect.get('ci_low')}-{effect.get('ci_high')}"
            p_value = f", P{effect.get('p_operator', '')}{effect.get('p_value')}" if effect.get("p_value") else ""
            fdr = f", FDR={effect.get('fdr')}" if effect.get("fdr") else ""
            effect_text = (
                f"Effect: {effect.get('exposure_candidates', '')} -> "
                f"{effect.get('disease') or ', '.join(source.disease_groups)}; "
                f"{effect.get('measure', '')}={effect.get('estimate', '')}{ci}{p_value}{fdr}."
            )
        block = (
            f"[Source {source.rank}]\n"
            f"Evidence group: {source.source_group_label}\n"
            f"Title: {source.title}\n"
            f"Source label: {source.source_label}\n"
            f"PMID: {source.pmid}; PMCID: {source.pmcid}; DOI: {source.doi}\n"
            f"URL: {source.source_url}\n"
            f"{effect_text}\n"
            f"Evidence snippet: {source.text[:650]}\n"
        )
        if used + len(block) > max_context_chars:
            break
        blocks.append(block)
        used += len(block)

    graph_blocks = [
        f"- {path.get('exposure')} -> {path.get('disease')}: "
        f"{path.get('measure')}={path.get('estimate')}; dataset={path.get('dataset')}; PMID={path.get('pmid')}"
        for path in (graph_paths or [])[:8]
    ]
    graph_text = "\n".join(graph_blocks) or "无匹配的结构化图谱路径。"
    lightrag_text = (lightrag_context or "无 LightRAG 补充上下文。")[:1800]
    return (
        "你是泌尿系统疾病暴露组学 RAG 助手。必须使用中文回答，并且只能使用给定证据。\n"
        "必须将“我的本地 UrologicalExpomics 数据”与“已发表论文/全文表格证据”分开讨论。\n"
        "不得编造 HR、OR、RR、95% CI、P 值、FDR、PMID、PMCID 或 DOI。\n"
        "若证据来自机器抽取的表格或效应量，需提醒用户正式写作前核对原始来源。\n"
        "引用格式使用 [Source n]。\n\n"
        f"用户问题：\n{query}\n\n"
        "检索证据：\n"
        + "\n".join(blocks)
        + "\n\n结构化 Neo4j 路径：\n"
        + graph_text
        + "\n\nLightRAG 图检索上下文：\n"
        + lightrag_text
        + "\n\n回答结构：\n"
        "1. 我的本地 UrologicalExpomics 数据\n"
        "2. 已发表论文/全文表格证据\n"
        "3. 两类证据是否一致，以及如何解释\n"
        "4. 注意事项和引用来源\n"
    )


def call_vllm_chat(
    prompt: str,
    temperature: float,
    *,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    base_url = (base_url or VLLM_BASE_URL).rstrip("/")
    model = model or VLLM_MODEL
    api_key = api_key or VLLM_API_KEY
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "你是中文生物医学 RAG 助手。只根据给定证据回答；不确定时必须明确说明。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 450,
    }
    response = requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=LLM_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise RuntimeError(f"{error}; body={response.text[:1000]}") from error
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def vllm_status() -> dict[str, Any]:
    try:
        response = requests.get(
            f"{VLLM_BASE_URL.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
            timeout=5,
        )
        response.raise_for_status()
        return {"available": True, "base_url": VLLM_BASE_URL, "model": VLLM_MODEL, "models": response.json()}
    except Exception as error:
        return {"available": False, "base_url": VLLM_BASE_URL, "model": VLLM_MODEL, "error": str(error)}
