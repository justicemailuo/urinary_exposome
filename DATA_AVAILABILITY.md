# Data Availability

This project separates code, reproducible data-generation scripts, and redistributable data artifacts.

## Recommended public data release

For GitHub Releases, the recommended public data package should include structured, tabular, source-linked evidence rather than raw copyrighted text. In particular, the release package may include:

- Local UrologicalExpomics effect-estimate tables.
- High-confidence abstract-level effect-estimate tables, with long text snippets removed.
- High-confidence full-text/table effect-estimate tables, with long text snippets removed.
- PubMed metadata tables, with abstracts removed.
- Search configuration and a manifest describing record counts and file provenance.

The release package should not include:

- API keys, passwords, or `.env` files.
- Raw PubMed abstracts as a bulk redistributed corpus.
- Full-text XML or article text unless the license has been checked.
- Chroma, Neo4j, or LightRAG database directories.
- Local Python environments or model caches.

## Why raw literature text is not bundled

PubMed abstracts and many PMC full-text articles can be publicly accessible without being freely redistributable as a bulk dataset. To reduce copyright and license risk, this repository provides scripts to retrieve literature metadata and rebuild the text corpus from source. Users who need full-text redistribution should restrict downloads to appropriately licensed sources, such as the PMC Open Access Subset, and preserve the applicable license information.

## Rebuilding data locally

Users can rebuild the literature corpus with:

```powershell
cd source\urinary_exposome_rag

python .\scripts\fetch_pubmed.py --config .\config\search_plan.json --max-per-query 8
python .\scripts\build_rag_corpus.py --input .\data\raw\pubmed_records.json --output-dir .\data\rag
python .\scripts\fetch_fulltext_and_tables.py --input .\data\raw\pubmed_records.json --output-dir .\data\fulltext
python .\scripts\index_fulltext_outputs.py --records .\data\raw\pubmed_records.json --fulltext-dir .\data\fulltext
python .\scripts\extract_effect_estimates.py --input .\data\raw\pubmed_records.json --output-dir .\data\effects
python .\scripts\extract_fulltext_effects.py --structured-dir .\data\fulltext\structured --output-dir .\data\effects
python .\scripts\build_fulltext_rag_corpus.py --structured-dir .\data\fulltext\structured --output-dir .\data\rag
```

## Creating a public data package

From the repository root:

```powershell
python .\source\urinary_exposome_rag\scripts\create_public_data_release.py
```

The script creates a sanitized ZIP file under `releases/`. This ZIP can be uploaded manually to a GitHub Release.
