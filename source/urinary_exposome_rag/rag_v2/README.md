# Urological Exposomics RAG v2

This prototype separates the UI, retrieval backend, vector database, embedding model, and local LLM.

```text
User
  -> Streamlit 4-page app
  -> FastAPI RAG backend
  -> Chroma vector store
  -> bge-m3 embeddings
  -> Qwen2.5-7B-Instruct through vLLM
```

The same FastAPI backend can also be called by Shiny with `httr2::req_perform()` or `httr::POST()`.

## Neo4j + LightRAG graph layer

The existing Chroma search remains the primary cited-document retriever. A second,
optional graph layer adds the deterministic path:

```text
Exposure -> EffectEstimate -> Disease
                    |-> Dataset
                    |-> Publication
```

Copy `.env.example` to `.env`, set a strong `NEO4J_PASSWORD`, and start Neo4j:

```powershell
docker compose --env-file .env -f docker-compose.graph.yml up -d
python scripts/build_knowledge_graph.py --reset
```

The isolated Neo4j instance is available at `http://127.0.0.1:7475` and
`bolt://127.0.0.1:7688`; ports 7474/7687 remain available to `pca_prot_rag`.

Graph endpoints:

- `POST /api/graph/index`: build the deterministic graph from the existing effect CSV files.
- `POST /api/graph/search`: return auditable exposure-effect-disease paths.
- `POST /api/lightrag/index`: optionally extract additional entities/relations with LightRAG.

LightRAG is disabled by default because indexing invokes the configured vLLM. Enable it
only after Neo4j and vLLM are healthy:

```powershell
$env:RAG_LIGHTRAG_ENABLED="true"
python scripts/build_lightrag_index.py --limit 100
```

`POST /api/chat` accepts `use_graph` and `use_lightrag`. Graph failures are returned in
`graph_error` or `lightrag_error` and do not disable the existing Chroma answer.

## Pages

1. `RAG Chat`: ask natural-language questions and receive answers with citations.
2. `Evidence Search`: inspect retrieved evidence snippets.
3. `HR / OR / RR Explorer`: focus on structured effect estimates.
4. `System Status`: check FastAPI, Chroma, bge-m3, vLLM status and rebuild the index.

## Install

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\rag_v2"
python -m pip install -r .\requirements.txt
```

## Build Chroma Index

First run downloads `BAAI/bge-m3`, so it can take several minutes.

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\rag_v2"
python .\scripts\build_chroma_index.py --reset --batch-size 64
```

For a quick smoke test:

```powershell
python .\scripts\build_chroma_index.py --reset --batch-size 16 --limit 512
```

## Start FastAPI

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\rag_v2"
.\scripts\run_fastapi.ps1
```

FastAPI runs at:

```text
http://127.0.0.1:8890
```

## Start Streamlit

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\rag_v2"
.\scripts\run_streamlit.ps1
```

Streamlit runs at:

```text
http://127.0.0.1:8501
```

## One-Command Startup

Local machine only:

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\rag_v2"
.\scripts\run_all_local.ps1
```

LAN access:

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\rag_v2"
.\scripts\run_all_lan.ps1
```

Stop both services:

```powershell
.\scripts\stop_all.ps1
```

## Command-line Query

With FastAPI running, query the same retrieval, graph, and generation pipeline without opening Streamlit:

```powershell
python .\scripts\query_rag.py "UKB 中 PM2.5 与肾癌的 HR 结果是什么？"
python .\scripts\query_rag.py "PFAS 与肾癌有什么证据？" --no-llm --no-graph
python .\scripts\query_rag.py "哪些暴露与膀胱癌相关？" --lightrag --json
```

The Streamlit chat page now exposes separate switches for Neo4j paths and LightRAG context. Both are optional: unavailable services produce a visible warning while Chroma retrieval continues.

## Start vLLM Qwen2.5-7B-Instruct

Run this in WSL2 Ubuntu or on a Linux GPU server:

```powershell
wsl -d Ubuntu-22.04 -- bash -lc "cd '/mnt/c/Users/Administrator/Documents/New project 2/urinary_exposome_rag/rag_v2' && bash scripts/setup_wsl_vllm_qwen.sh"
```

Then start vLLM:

```bash
cd "/mnt/c/Users/Administrator/Documents/New project 2/urinary_exposome_rag/rag_v2"
bash scripts/start_wsl_vllm_qwen25_7b.sh
```

On this 16 GB GPU setup, the script serves `Qwen/Qwen2.5-7B-Instruct-AWQ` as
`Qwen/Qwen2.5-7B-Instruct`, disables FlashInfer sampling, uses Triton AWQ, and
sets `--gpu-memory-utilization 0.65` to avoid KV-cache OOM on WSL2.

The backend expects the OpenAI-compatible vLLM API at:

```text
http://127.0.0.1:8001/v1
```

If vLLM is not running, the backend returns a retrieval-only answer and keeps sources/citations.

## Shiny Call Pattern

The existing Shiny app can call this backend the same way it calls the older RAG API:

```r
payload <- list(
  query = "UKB 中 PM2.5 和肾癌的 HR 结果是什么？请把我的本地数据和论文证据分开回答。",
  top_k = 8,
  use_llm = TRUE,
  filters = list(
    source = "all",
    exposure_domain = "environmental_pollution",
    disease_group = "renal_cancer",
    effects_only = TRUE,
    table_only = FALSE,
    chinese_only = FALSE
  )
)

response <- httr2::request("http://127.0.0.1:8890/api/chat") |>
  httr2::req_method("POST") |>
  httr2::req_body_json(payload, auto_unbox = TRUE) |>
  httr2::req_perform()

result <- httr2::resp_body_json(response)
```
