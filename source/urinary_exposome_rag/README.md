# Urinary Exposome RAG Knowledge Base

This project builds a source-linked RAG knowledge base for relationships among:

- Urinary system diseases
- Lifestyle exposures
- Environmental pollution and climate/built-environment exposures
- Baseline diseases and comorbidities

The pipeline uses PubMed/NCBI E-utilities instead of fragile page scraping.

## Outputs

- `data/raw/pubmed_records.json`: deduplicated PubMed records with metadata.
- `data/raw/search_log.json`: search terms, hit counts, and retrieved PMIDs.
- `data/tables/pubmed_records.csv`: spreadsheet-friendly literature table.
- `data/rag/rag_chunks.jsonl`: RAG-ready chunks with metadata.
- `data/rag/rag_documents.md`: human-readable corpus preview.
- `scripts/retrieve_tfidf.py`: local retrieval smoke test before vector database ingestion.
- `data/effects/effect_estimates_abstract_extracted.csv`: candidate HR/OR/RR values extracted from titles and abstracts.
- `data/fulltext/structured/*.json`: structured PMC full text parsed from Europe PMC/PMC XML.
- `data/fulltext/xml_tables/*.csv`: extracted tables from full-text XML.
- `data/fulltext/fulltext_manifest_indexed.csv`: full-text availability manifest.
- `data/effects/effect_estimates_fulltext_extracted.csv`: candidate HR/OR/RR values from full text and tables.
- `data/effects/effect_estimates_fulltext_high_confidence.csv`: full-text/table estimates with confidence intervals.
- `data/rag/rag_fulltext_chunks.jsonl`: RAG chunks from full text sections and tables.

## Quick Start

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag"
python .\scripts\fetch_pubmed.py --config .\config\search_plan.json --max-per-query 8
python .\scripts\build_rag_corpus.py --input .\data\raw\pubmed_records.json --output-dir .\data\rag
python .\scripts\extract_effect_estimates.py --input .\data\raw\pubmed_records.json --output-dir .\data\effects
python .\scripts\fetch_fulltext_and_tables.py --input .\data\raw\pubmed_records.json --output-dir .\data\fulltext
python .\scripts\index_fulltext_outputs.py --records .\data\raw\pubmed_records.json --fulltext-dir .\data\fulltext
python .\scripts\extract_fulltext_effects.py --structured-dir .\data\fulltext\structured --output-dir .\data\effects
python .\scripts\build_fulltext_rag_corpus.py --structured-dir .\data\fulltext\structured --output-dir .\data\rag
python .\scripts\retrieve_tfidf.py --query "PM2.5 and chronic kidney disease" --top-k 5
```

## Interactive RAG Chat

Start the local chat app:

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag"
python .\scripts\rag_chat_server.py --host 127.0.0.1 --port 8765
```

Open:

```text
http://127.0.0.1:8765
```

The app indexes:

- `data/rag/rag_chunks.jsonl`
- `data/rag/rag_fulltext_chunks.jsonl`
- `data/effects/effect_estimates_high_confidence.csv`
- `data/effects/effect_estimates_fulltext_high_confidence.csv`

It supports filters for exposure domain, disease group, abstract/full-text
source, table-only retrieval, effect-estimate-only retrieval, and China/Chinese
population candidates.

## UrologicalExpomics Shiny Integration

The existing Shiny app at:

```text
C:\Users\Administrator\Desktop\UrologicalExpomics
```

has been integrated with this RAG backend. A new `RAG Assistant` tab was added
to `app.R`. It calls:

```text
http://127.0.0.1:8765/api/chat
```

The local UrologicalExpomics result files were exported into RAG-readable effect
tables:

- `data/effects/effect_estimates_urological_expomics_local.csv`
- `data/effects/effect_estimates_urological_expomics_local.jsonl`

These include UKB exogenous XWAS, NHANES endogenous XWAS, and WeEndPd preliminary
results with HR/OR, 95% CI, P value, FDR, exposure, disease, dataset, and
population metadata.

Start both services:

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag"
python .\scripts\rag_chat_server.py --host 127.0.0.1 --port 8765
```

In another terminal:

```powershell
Rscript "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\scripts\run_shiny_urological_expomics.R"
```

Open:

```text
http://127.0.0.1:8787
```

The default RAG question in Shiny is set to retrieve UKB PM2.5 and renal cancer
HR results, with filters defaulting to effect estimates, environmental
pollution, and renal cancer.

### Optional Local Qwen 7B Generation

The RAG backend can optionally call a local Ollama model such as Qwen 7B after
retrieving evidence. Retrieval still comes from the curated RAG index; the model
only rewrites/synthesizes the retrieved evidence.

Install Ollama, then pull a Qwen 7B-class model, for example:

```powershell
ollama pull qwen2.5:7b-instruct
```

Start the RAG backend with Ollama generation:

```powershell
cd "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag"
python .\scripts\rag_chat_server.py --host 127.0.0.1 --port 8765 --llm-provider ollama --llm-model qwen2.5:7b-instruct --llm-base-url http://127.0.0.1:11434
```

Then start Shiny as usual:

```powershell
Rscript "C:\Users\Administrator\Documents\New project 2\urinary_exposome_rag\scripts\run_shiny_urological_expomics.R"
```

In the Shiny `RAG Assistant` tab, enable:

```text
Use local Qwen / Ollama model
```

If Ollama or the model is not running, the backend automatically falls back to
the deterministic retrieval answer.

Optional environment variables:

- `NCBI_EMAIL`: recommended by NCBI for API contact.
- `NCBI_API_KEY`: increases NCBI rate limits if you have an API key.

## Expand the Knowledge Base

Edit `config/search_plan.json`:

- Add exposure terms under `exposure_domains`.
- Add diseases under `disease_groups`.
- Increase `--max-per-query` for a larger corpus.
- Change `date_range` if you only want recent literature.

Recommended next scaling pass:

```powershell
python .\scripts\fetch_pubmed.py --config .\config\search_plan.json --max-per-query 50
python .\scripts\build_rag_corpus.py --input .\data\raw\pubmed_records.json --output-dir .\data\rag
```

## Import to a RAG System

Use `data/rag/rag_chunks.jsonl` as the canonical ingestion file. Embed the `text`
field and keep the remaining fields as metadata filters. Useful filters are
`exposure_domains`, `disease_groups`, `publication_year`, `pmid`, and
`source_url`.

For full-text/table RAG, also ingest `data/rag/rag_fulltext_chunks.jsonl`.
Use `fulltext_location=table` to filter table-derived chunks.

## Effect Estimate Table

`data/effects/effect_estimates_abstract_extracted.csv` is an abstract-level,
machine-extracted evidence table. It captures HR, OR, RR, IRR, 95% CI, nearby
P values, candidate exposure terms, disease group labels, and source links when
these values appear in PubMed titles or abstracts.

`data/effects/effect_estimates_fulltext_extracted.csv` extends this to PMC
full text and XML tables. `source_location=xml_table` means the candidate value
came from a parsed table rather than prose.

This file is useful for triage and RAG retrieval, but it is not a final
meta-analysis table. Many papers report detailed exposure-specific estimates
only in full-text tables or supplementary files, so rows marked
`needs_manual_check=yes` should be manually verified before formal reporting.

## Full Text and PDF Notes

The pipeline prioritizes PMC/Europe PMC structured XML because it preserves
article sections and tables better than PDF text extraction. PDF downloading is
best-effort: some PMC PDF links are generated through a browser download page
and do not return a PDF binary to a plain script request. In those cases, use
the XML-derived tables in `data/fulltext/xml_tables/` as the primary source.

## RAG Metadata Fields

Each JSONL chunk contains:

- `chunk_id`
- `text`
- `source_type`
- `source_url`
- `pmid`
- `doi`
- `title`
- `journal`
- `publication_year`
- `exposure_domains`
- `disease_groups`
- `query_labels`
- `mesh_terms`
- `keywords`
