# UrologicalExpomics Public Data Package

This package contains sanitized structured data for the UrologicalExpomics RAG workflow.

Included files:

- `data/tables/pubmed_records_metadata.csv`: PubMed metadata with abstracts removed.
- `data/effects/effect_estimates_high_confidence.csv`: high-confidence abstract-level effect estimates with snippets removed.
- `data/effects/effect_estimates_fulltext_high_confidence.csv`: high-confidence full-text/table effect estimates with snippets removed.
- `data/effects/effect_estimates_urological_expomics_local.csv`: local UrologicalExpomics effect-estimate records with snippets removed.
- `config/search_plan.json`: PubMed search configuration.
- `manifest.json`: package metadata and row counts.

This package is intended for evidence navigation and RAG index rebuilding. Machine-extracted effect estimates should be checked against the original sources before formal scientific interpretation.
