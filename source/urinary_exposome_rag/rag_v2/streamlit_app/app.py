from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


API_BASE_URL = os.environ.get("RAG_API_BASE_URL", "http://127.0.0.1:8890")


st.set_page_config(
    page_title="Urological Exposomics RAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


def api_post(path: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_get(path: str, timeout: int = 20) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def filters(prefix: str = "") -> dict[str, Any]:
    source = st.sidebar.selectbox(
        "Evidence source",
        options=[
            ("All", "all"),
            ("My local data", "local_data"),
            ("Literature only", "literature"),
            ("Effect estimates only", "effects"),
            ("Full text / tables", "fulltext"),
            ("Abstracts", "abstract"),
        ],
        format_func=lambda item: item[0],
        key=f"{prefix}source",
    )[1]
    exposure_domain = st.sidebar.selectbox(
        "Exposure domain",
        options=[
            ("All", "all"),
            ("Lifestyle", "lifestyle"),
            ("Environmental pollution", "environmental_pollution"),
            ("Climate / built environment", "climate_built_environment"),
            ("Baseline disease", "baseline_disease"),
            ("Local exposomics", "local_exposomics"),
        ],
        format_func=lambda item: item[0],
        key=f"{prefix}exposure",
    )[1]
    disease_group = st.sidebar.selectbox(
        "Disease group",
        options=[
            ("All", "all"),
            ("Renal cancer", "renal_cancer"),
            ("Bladder cancer", "bladder_cancer"),
            ("Chronic kidney disease", "chronic_kidney_disease"),
            ("Acute kidney injury", "acute_kidney_injury"),
            ("Urolithiasis", "urolithiasis"),
            ("Urinary tract infection", "urinary_tract_infection"),
            ("Lower urinary tract symptoms", "lower_urinary_tract_symptoms"),
        ],
        format_func=lambda item: item[0],
        key=f"{prefix}disease",
    )[1]
    return {
        "source": source,
        "exposure_domain": exposure_domain,
        "disease_group": disease_group,
        "effects_only": st.sidebar.checkbox("Only HR / OR / RR", value=True, key=f"{prefix}effects"),
        "table_only": st.sidebar.checkbox("Only table-derived evidence", value=False, key=f"{prefix}table"),
        "chinese_only": st.sidebar.checkbox("China / Chinese candidates", value=False, key=f"{prefix}china"),
    }


def source_dataframe(sources: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source in sources:
        effect = source.get("effect") or {}
        rows.append(
            {
                "rank": source.get("rank"),
                "score": source.get("score"),
                "group": source.get("source_group_label"),
                "title": source.get("title"),
                "source": source.get("source_label"),
                "measure": effect.get("measure"),
                "estimate": effect.get("estimate"),
                "ci_low": effect.get("ci_low"),
                "ci_high": effect.get("ci_high"),
                "p": effect.get("p_value"),
                "fdr": effect.get("fdr"),
                "exposure": effect.get("exposure_candidates"),
                "disease": effect.get("disease") or "; ".join(source.get("disease_groups", [])),
                "pmid": source.get("pmid"),
                "pmcid": source.get("pmcid"),
            }
        )
    return pd.DataFrame(rows)


def render_sources(sources: list[dict[str, Any]]) -> None:
    for source in sources:
        effect = source.get("effect") or {}
        title = source.get("title") or source.get("source_label")
        with st.expander(f"#{source.get('rank')} {source.get('source_group_label')} | {title}"):
            st.caption(f"Score {source.get('score')} | {source.get('source_label')}")
            if effect:
                ci = ""
                if effect.get("ci_low") and effect.get("ci_high"):
                    ci = f" (95% CI {effect.get('ci_low')}-{effect.get('ci_high')})"
                p_value = f", P{effect.get('p_operator', '')}{effect.get('p_value')}" if effect.get("p_value") else ""
                fdr = f", FDR={effect.get('fdr')}" if effect.get("fdr") else ""
                st.markdown(f"**{effect.get('measure')}={effect.get('estimate')}{ci}{p_value}{fdr}**")
            st.write(source.get("text", ""))
            if source.get("source_url"):
                st.link_button("Open source", source["source_url"])


def page_chat() -> None:
    st.title("RAG Chat")
    st.caption("FastAPI -> Chroma/bge-m3 -> Qwen2.5-7B-Instruct via vLLM, with retrieval fallback.")
    query = st.text_area(
        "Question",
        value="UKB 中 PM2.5 和肾癌的 HR 结果是什么？请把我的本地数据和论文证据分开回答。",
        height=110,
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        top_k = st.slider("Top K", min_value=3, max_value=30, value=8)
    with col2:
        use_llm = st.toggle("Use vLLM / Qwen", value=True)
    with col3:
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)

    option_col1, option_col2 = st.columns(2)
    with option_col1:
        use_graph = st.toggle(
            "Use Neo4j graph",
            value=True,
            help="Retrieve auditable exposure-effect-disease paths.",
        )
    with option_col2:
        use_lightrag = st.toggle(
            "Use LightRAG context",
            value=False,
            help="Requires RAG_LIGHTRAG_ENABLED=true and an existing LightRAG index.",
        )

    current_filters = filters("chat_")
    if st.button("Ask", type="primary", use_container_width=True):
        payload = {
            "query": query,
            "top_k": top_k,
            "use_llm": use_llm,
            "temperature": temperature,
            "use_graph": use_graph,
            "use_lightrag": use_lightrag,
            "filters": current_filters,
        }
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                result = api_post("/api/chat", payload)
            except Exception as error:
                st.error(f"RAG request failed: {error}")
                return
        st.session_state["last_chat_result"] = result

    result = st.session_state.get("last_chat_result")
    if result:
        st.info(
            f"Sources: {len(result.get('sources', []))} | "
            f"local: {result.get('source_groups', {}).get('local_data', 0)} | "
            f"literature: {result.get('source_groups', {}).get('literature', 0)} | "
            f"{result.get('elapsed_ms')} ms | LLM used: {result.get('llm_used')}"
        )
        if result.get("llm_error"):
            st.warning(result["llm_error"])
        if result.get("graph_error"):
            st.warning(f"Neo4j graph unavailable: {result['graph_error']}")
        if result.get("lightrag_error"):
            st.warning(f"LightRAG unavailable: {result['lightrag_error']}")
        st.markdown(result.get("answer", ""))
        if result.get("graph_paths"):
            with st.expander(f"Neo4j evidence paths ({len(result['graph_paths'])})"):
                st.dataframe(pd.DataFrame(result["graph_paths"]), use_container_width=True, hide_index=True)
        if result.get("lightrag_context"):
            with st.expander("LightRAG retrieved context"):
                st.text(result["lightrag_context"])
        st.divider()
        st.subheader("Evidence")
        render_sources(result.get("sources", []))


def page_search() -> None:
    st.title("Evidence Search")
    query = st.text_input("Search query", value="PFAS renal cancer RR 95% CI")
    top_k = st.slider("Top K", min_value=3, max_value=30, value=12)
    current_filters = filters("search_")
    if st.button("Search evidence", type="primary", use_container_width=True):
        payload = {
            "query": query,
            "top_k": top_k,
            "use_llm": False,
            "filters": current_filters,
        }
        with st.spinner("Searching Chroma..."):
            try:
                result = api_post("/api/chat", payload)
            except Exception as error:
                st.error(f"Search failed: {error}")
                return
        st.session_state["last_search_result"] = result

    result = st.session_state.get("last_search_result")
    if result:
        st.metric("Retrieved sources", len(result.get("sources", [])))
        df = source_dataframe(result.get("sources", []))
        st.dataframe(df, use_container_width=True, hide_index=True)
        render_sources(result.get("sources", []))


def page_effects() -> None:
    st.title("HR / OR / RR Explorer")
    query = st.text_input("Effect query", value="PM2.5 renal cancer HR")
    top_k = st.slider("Rows", min_value=5, max_value=30, value=20)
    current_filters = filters("effects_")
    current_filters["effects_only"] = True
    if st.button("Retrieve effect estimates", type="primary", use_container_width=True):
        payload = {
            "query": query,
            "top_k": top_k,
            "use_llm": False,
            "filters": current_filters,
        }
        with st.spinner("Retrieving structured effects..."):
            try:
                result = api_post("/api/chat", payload)
            except Exception as error:
                st.error(f"Effect retrieval failed: {error}")
                return
        st.session_state["last_effect_result"] = result

    result = st.session_state.get("last_effect_result")
    if result:
        df = source_dataframe(result.get("sources", []))
        st.dataframe(df, use_container_width=True, hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download CSV", data=csv, file_name="rag_effect_estimates.csv", mime="text/csv")


def page_system() -> None:
    st.title("System Status")
    st.caption("FastAPI, Chroma index, bge-m3 embedding model, and vLLM/Qwen status.")
    if st.button("Refresh status", use_container_width=True):
        st.session_state.pop("system_status", None)
    if "system_status" not in st.session_state:
        try:
            st.session_state["system_status"] = api_get("/health")
        except Exception as error:
            st.error(f"Cannot reach FastAPI backend at {API_BASE_URL}: {error}")
            return
    status = st.session_state["system_status"]
    st.json(status)

    st.subheader("Build / rebuild Chroma index")
    st.write("This calls the FastAPI backend to index existing RAG JSONL and effect CSV files into Chroma with bge-m3 embeddings.")
    col1, col2 = st.columns(2)
    with col1:
        reset = st.checkbox("Reset collection before indexing", value=False)
    with col2:
        batch_size = st.number_input("Batch size", min_value=8, max_value=256, value=64, step=8)
    limit = st.number_input("Optional document limit for smoke test", min_value=0, max_value=100000, value=0, step=100)
    if st.button("Build index", type="primary", use_container_width=True):
        with st.spinner("Building Chroma index. First run may download BAAI/bge-m3 and take several minutes..."):
            try:
                result = requests.post(
                    f"{API_BASE_URL}/api/index",
                    params={
                        "reset": reset,
                        "batch_size": int(batch_size),
                        "limit": int(limit) if int(limit) > 0 else None,
                    },
                    timeout=3600,
                )
                result.raise_for_status()
                st.success("Index build completed.")
                st.json(result.json())
                st.session_state.pop("system_status", None)
            except Exception as error:
                st.error(f"Index build failed: {error}")

    st.subheader("Backend URLs")
    st.code(
        f"FastAPI: {API_BASE_URL}\n"
        f"Health: {API_BASE_URL}/health\n"
        f"Chat: {API_BASE_URL}/api/chat\n"
        "vLLM default: http://127.0.0.1:8001/v1",
        language="text",
    )


def main() -> None:
    st.sidebar.title("Urological RAG v2")
    st.sidebar.caption(f"API: {API_BASE_URL}")
    page = st.sidebar.radio(
        "Page",
        ["RAG Chat", "Evidence Search", "HR / OR / RR Explorer", "System Status"],
    )
    st.sidebar.divider()

    if page == "RAG Chat":
        page_chat()
    elif page == "Evidence Search":
        page_search()
    elif page == "HR / OR / RR Explorer":
        page_effects()
    else:
        page_system()


if __name__ == "__main__":
    main()
