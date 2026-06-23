# UrologicalExpomics RAG

This repository provides a retrieval-augmented generation (RAG) workflow for integrating local urological exposomics results with biomedical literature evidence. The system was designed to support evidence navigation, hypothesis exploration, and source-grounded interpretation of exposure–disease associations across urinary system outcomes.

The workflow combines PubMed literature retrieval, structured full-text/table parsing, machine-extracted epidemiological effect estimates, local UrologicalExpomics association results, vector-based semantic search, Neo4j graph representation, and an interactive question-answering interface.

## Overview

The system separates evidence into two major sources:

1. **Local UrologicalExpomics evidence**: structured exposure–disease association results derived from local analyses, including effect estimates, confidence intervals, P values, false discovery rate values, dataset labels, disease outcomes, and exposure metadata.
2. **Published literature evidence**: PubMed records, abstracts, available full-text XML documents, extracted tables, and candidate epidemiological effect estimates from published studies.

The web interface allows users to ask natural-language questions and retrieve source-linked answers. Retrieved evidence can be summarized by an OpenAI-compatible language model, while the underlying records, effect estimates, and graph paths remain inspectable.

## Current Evidence Base

The current indexed evidence base contains:

| Evidence component | Count |
|---|---:|
| PubMed search queries | 28 |
| Initial PubMed candidate hits | 227,260 |
| Deduplicated PubMed records included in the knowledge base | 129 |
| Publication year range | 2011–2026 |
| Structured full-text/XML articles | 34 |
| Parsed XML table files | 62 |
| Abstract-level RAG text chunks | 268 |
| Full-text-level RAG text chunks | 1,389 |
| Candidate effect estimates extracted from abstracts | 67 |
| High-confidence abstract-level effect estimates | 37 |
| Candidate effect estimates extracted from full text/tables | 475 |
| High-confidence full-text/table effect estimates | 244 |
| Local UrologicalExpomics effect-estimate records | 49,769 |
| Neo4j graph nodes | >50,000 |

The initial PubMed candidate hits represent the total number of records returned across all search queries. The final literature knowledge base currently includes 129 deduplicated PubMed records with parsed metadata. Machine-extracted effect estimates are intended for evidence navigation and should be manually verified against the original publications before formal interpretation.

## Workflow

The workflow consists of seven main steps:

1. Define disease- and exposure-oriented PubMed search queries.
2. Retrieve and deduplicate PubMed records using NCBI/PubMed metadata.
3. Parse titles, abstracts, publication years, PMIDs, DOIs, MeSH terms, and source URLs.
4. Retrieve available PMC/Europe PMC full-text XML documents and extract structured sections and tables.
5. Extract candidate epidemiological estimates, including HR, OR, RR, confidence intervals, and P values.
6. Convert local UrologicalExpomics association results and literature-derived records into RAG-ready documents.
7. Build semantic vector indices and a Neo4j graph linking exposures, effect estimates, diseases, datasets, and publications.

## System Architecture

```text
User question
  -> Streamlit web interface
  -> FastAPI backend
  -> Chroma/bge-m3 semantic retrieval
  -> optional Neo4j graph-path retrieval
  -> optional LightRAG context retrieval
  -> optional OpenAI-compatible language model
  -> source-grounded answer
```

The response prompt requires the generated answer to distinguish local UrologicalExpomics evidence from published literature evidence and to avoid unsupported effect sizes, identifiers, or citations.

## Repository Structure

```text
source/urinary_exposome_rag/
  config/                 PubMed search configuration
  scripts/                Literature retrieval, full-text parsing, and effect extraction
  data/                   Generated data, RAG chunks, and effect-estimate tables
  rag_v2/                 FastAPI + Streamlit + Chroma + Neo4j RAG application

source/urinary_exposome_rag/rag_v2/
  backend/                FastAPI backend and retrieval logic
  streamlit_app/          Interactive web application
  scripts/                Index building, service startup, and command-line query helpers
  docker-compose.graph.yml Neo4j graph service
```

## Quick Start

Enter the RAG v2 directory:

```powershell
cd "H:\UrologicalExpomics_RAG_bundle_20260611\source\urinary_exposome_rag\rag_v2"
```

Install dependencies if needed:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

Start the FastAPI backend and Streamlit interface:

```powershell
.\scripts\run_all_local.ps1
```

Open:

```text
http://127.0.0.1:8501
```

## Using the Bundled Public Data

The repository includes a sanitized `source/urinary_exposome_rag/public_data/` directory. It contains structured effect-estimate tables and PubMed metadata with abstracts and long snippets removed. This allows new users to run the workflow without access to the original local machine data.

After cloning the repository, install the bundled public data:

```powershell
python .\source\urinary_exposome_rag\scripts\install_public_data.py
```

Then build the Chroma index:

```powershell
cd .\source\urinary_exposome_rag\rag_v2
.\.venv\Scripts\python.exe .\scripts\build_chroma_index.py --reset --batch-size 64
```

For a quick smoke test:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_chroma_index.py --reset --batch-size 16 --limit 500
```

## Optional Neo4j Graph Layer

Start Neo4j:

```powershell
docker compose --env-file .env -f .\docker-compose.graph.yml up -d
```

Build or rebuild the graph:

```powershell
.\.venv\Scripts\python.exe .\scripts\build_knowledge_graph.py --reset
```

The graph layer provides auditable paths such as:

```text
Exposure -> EffectEstimate -> Disease
                 |-> Dataset
                 |-> Publication
```

## Optional Language Model Configuration

The application can call any OpenAI-compatible chat completions endpoint. For example:

```env
VLLM_BASE_URL=https://yunwu.ai/v1
VLLM_MODEL=gpt-5.4-nano:stable
VLLM_API_KEY=your_api_token_here
```

Only the base URL should be provided in `VLLM_BASE_URL`; the application appends `/chat/completions` automatically.

Do not upload `.env` files or API keys to GitHub.

### Demo API quota

For a hosted public demo, the server can provide a limited number of free model calls using the server-side API key. By default, each browser/client receives 10 successful demo model calls. After the quota is exhausted, the Streamlit UI asks the user to enter their own OpenAI-compatible API key.

Configure this in `source/urinary_exposome_rag/rag_v2/.env`:

```env
RAG_DEMO_API_ENABLED=true
RAG_DEMO_API_FREE_CALLS=10
RAG_DEMO_API_QUOTA_FILE=./data/demo_api_quota.json
```

The server-side demo API key remains in `.env` and is never sent to the browser. User-provided API keys are sent only to the backend for the current request and are not stored by the app. For a serious public service, add HTTPS, login, rate limiting, and abuse monitoring.

## Command-Line Query

After FastAPI is running, query the system from the command line:

```powershell
python .\scripts\query_rag.py "What evidence links PM2.5 to renal cancer?" --no-llm
```

Return the full JSON response:

```powershell
python .\scripts\query_rag.py "What evidence links PFAS to bladder cancer?" --json
```

## Public Data Release

The Git repository is code-first and does not commit raw generated databases, local model caches, or full-text corpora. For users who want a more ready-to-use starting point, a sanitized public data package can be distributed through GitHub Releases.

The recommended release package includes structured evidence tables and PubMed metadata while removing long copyrighted text fields such as abstracts and snippets. See `DATA_AVAILABILITY.md` for the rationale and detailed file list.

To create the release ZIP locally:

```powershell
python .\source\urinary_exposome_rag\scripts\create_public_data_release.py
```

The generated ZIP under `releases/` can be uploaded manually to a GitHub Release. Users can then download the ZIP, extract it into `source/urinary_exposome_rag/`, provide their own OpenAI-compatible model API key in `.env`, rebuild the Chroma index, and start the web application.

## Suggested Manuscript Description

A concise workflow description suitable for the Methods or Results section is:

> We developed a retrieval-augmented urological exposomics workflow that integrates local exposure–disease association results with PubMed-derived biomedical evidence. Across 28 predefined PubMed search queries, 227,260 candidate records were identified, from which 129 deduplicated PubMed records were incorporated into the current knowledge base. These records covered publications from 2011 to 2026. Structured full-text or XML data were available for 34 articles, yielding 62 parsed table files. The literature corpus was converted into 268 abstract-level and 1,389 full-text-level RAG text chunks. In parallel, 49,769 local UrologicalExpomics effect-estimate records were harmonized and indexed. The system combines semantic retrieval, graph-based exposure–effect–disease representation, and optional language-model generation to provide source-grounded answers that distinguish local analytical results from published literature evidence.

## GitHub Upload Notes

Recommended files to upload:

- Source code
- Search configuration files
- Documentation
- Safe configuration templates such as `.env.example`
- Small example data, if appropriate

Do not upload:

- `.env` files
- API keys or passwords
- Local Python environments
- Model caches
- Neo4j, Chroma, or LightRAG database directories
- Large generated raw/full-text data unless intentionally released

Example Git commands:

```powershell
git init
git add README.md .gitignore source/urinary_exposome_rag
git commit -m "Initial UrologicalExpomics RAG workflow"
git branch -M main
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```

## Research Use and Limitations

This system is intended for evidence navigation, hypothesis exploration, and literature-assisted interpretation. It is not a substitute for a systematic review, formal meta-analysis, or causal inference framework. Machine-extracted effect estimates from abstracts, full text, and tables should be manually checked against the original sources before use in formal scientific reporting.
