from __future__ import annotations

import hashlib
import re
import time
from functools import lru_cache
from typing import Any

from .config import (
    EFFECT_DIR,
    GRAPH_ENABLED,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    NEO4J_WORKSPACE,
)
from .data_loader import _load_effect_csv, split_semicolon
from .schemas import RagFilters


@lru_cache(maxsize=1)
def get_driver():
    if not GRAPH_ENABLED:
        raise RuntimeError("Neo4j graph support is disabled")
    if not NEO4J_PASSWORD:
        raise RuntimeError("NEO4J_PASSWORD is not configured")
    try:
        from neo4j import GraphDatabase
    except ImportError as error:
        raise RuntimeError("Neo4j driver is not installed; run pip install -r requirements.txt") from error
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


def graph_status() -> dict[str, Any]:
    if not GRAPH_ENABLED:
        return {"enabled": False, "available": False}
    try:
        with get_driver().session(database=NEO4J_DATABASE) as session:
            row = session.run(
                "MATCH (n {workspace: $workspace}) RETURN count(n) AS nodes",
                workspace=NEO4J_WORKSPACE,
            ).single()
        return {
            "enabled": True,
            "available": True,
            "workspace": NEO4J_WORKSPACE,
            "nodes": int(row["nodes"] if row else 0),
        }
    except Exception as error:
        return {"enabled": True, "available": False, "error": str(error)}


def _effect_rows(limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    effect_sources = (
        (EFFECT_DIR / "effect_estimates_high_confidence.csv", "abstract_effects"),
        (EFFECT_DIR / "effect_estimates_fulltext_high_confidence.csv", "fulltext_effects"),
        (EFFECT_DIR / "effect_estimates_urological_expomics_local.csv", "urological_expomics_local"),
    )
    documents: list[dict[str, Any]] = []
    for path, collection in effect_sources:
        remaining = None if limit is None else max(0, limit - len(documents))
        if remaining == 0:
            break
        loaded = _load_effect_csv(path, collection)
        documents.extend(loaded if remaining is None else loaded[:remaining])
    for doc in documents:
        metadata = doc["metadata"]
        if metadata.get("source_type") != "effect_estimate":
            continue
        exposures = split_semicolon(str(metadata.get("exposure_candidates", "")))
        if not exposures:
            exposures = split_semicolon(str(metadata.get("exposure_domains", "")))
        diseases = split_semicolon(str(metadata.get("disease", "")))
        if not diseases:
            diseases = split_semicolon(str(metadata.get("disease_groups", "")))
        if not exposures or not diseases:
            continue
        publication_key = str(metadata.get("pmid") or metadata.get("doi") or metadata.get("source_url") or "")
        if not publication_key:
            publication_key = f"local:{metadata.get('source_dataset') or metadata.get('collection')}"
        effect_key = hashlib.sha256(str(doc["id"]).encode("utf-8")).hexdigest()[:24]
        rows.append(
            {
                "workspace": NEO4J_WORKSPACE,
                "effect_id": effect_key,
                "document_id": str(doc["id"]),
                "exposures": exposures,
                "diseases": diseases,
                "measure": str(metadata.get("measure", "")),
                "estimate": str(metadata.get("estimate", "")),
                "ci_low": str(metadata.get("ci_low", "")),
                "ci_high": str(metadata.get("ci_high", "")),
                "p_value": str(metadata.get("p_value", "")),
                "fdr": str(metadata.get("fdr", "")),
                "dataset": str(metadata.get("source_dataset") or metadata.get("collection") or "unknown"),
                "source_group": str(metadata.get("source_group", "literature")),
                "publication_key": publication_key,
                "pmid": str(metadata.get("pmid", "")),
                "pmcid": str(metadata.get("pmcid", "")),
                "doi": str(metadata.get("doi", "")),
                "title": str(metadata.get("title", "")),
                "source_url": str(metadata.get("source_url", "")),
                "snippet": str(metadata.get("text", doc["document"]))[:1200],
            }
        )
        if limit and len(rows) >= limit:
            break
    return rows


def build_graph(reset: bool = False, batch_size: int = 250, limit: int | None = None) -> dict[str, Any]:
    started = time.time()
    rows = _effect_rows(limit)
    driver = get_driver()
    with driver.session(database=NEO4J_DATABASE) as session:
        if reset:
            session.run("MATCH (n {workspace: $workspace}) DETACH DELETE n", workspace=NEO4J_WORKSPACE).consume()
        for label in ("Exposure", "Disease", "EffectEstimate", "Dataset", "Publication"):
            session.run(
                f"CREATE CONSTRAINT {label.lower()}_workspace_key IF NOT EXISTS "
                f"FOR (n:{label}) REQUIRE (n.workspace, n.key) IS UNIQUE"
            ).consume()
        query = """
        UNWIND $rows AS row
        MERGE (fx:EffectEstimate {workspace: row.workspace, key: row.effect_id})
        SET fx.document_id=row.document_id, fx.measure=row.measure, fx.estimate=row.estimate,
            fx.ci_low=row.ci_low, fx.ci_high=row.ci_high, fx.p_value=row.p_value,
            fx.fdr=row.fdr, fx.source_group=row.source_group, fx.snippet=row.snippet
        MERGE (ds:Dataset {workspace: row.workspace, key: row.dataset})
        SET ds.name=row.dataset
        MERGE (pub:Publication {workspace: row.workspace, key: row.publication_key})
        SET pub.pmid=row.pmid, pub.pmcid=row.pmcid, pub.doi=row.doi,
            pub.title=row.title, pub.source_url=row.source_url
        MERGE (fx)-[:FROM_DATASET]->(ds)
        MERGE (fx)-[:REPORTED_IN]->(pub)
        FOREACH (exposure IN row.exposures |
          MERGE (ex:Exposure {workspace: row.workspace, key: toLower(exposure)})
          SET ex.name=exposure
          MERGE (ex)-[:HAS_EFFECT]->(fx)
        )
        FOREACH (disease IN row.diseases |
          MERGE (dis:Disease {workspace: row.workspace, key: toLower(disease)})
          SET dis.name=disease
          MERGE (fx)-[:ON_DISEASE]->(dis)
        )
        """
        for start in range(0, len(rows), batch_size):
            session.run(query, rows=rows[start : start + batch_size]).consume()
        counts = session.run(
            "MATCH (n {workspace: $workspace}) "
            "RETURN labels(n)[0] AS label, count(n) AS count ORDER BY label",
            workspace=NEO4J_WORKSPACE,
        ).data()
    return {
        "workspace": NEO4J_WORKSPACE,
        "effect_rows": len(rows),
        "counts": {row["label"]: row["count"] for row in counts},
        "elapsed_ms": int((time.time() - started) * 1000),
    }


def _query_terms(query: str) -> list[str]:
    return [term.lower() for term in re.findall(r"[A-Za-z0-9_.-]{2,}|[\u4e00-\u9fff]{2,}", query)][:12]


def search_graph(query: str, top_k: int, filters: RagFilters) -> list[dict[str, Any]]:
    terms = _query_terms(query)
    cypher = """
    MATCH (ex:Exposure)-[:HAS_EFFECT]->(fx:EffectEstimate)-[:ON_DISEASE]->(dis:Disease)
    MATCH (fx)-[:FROM_DATASET]->(ds:Dataset)
    MATCH (fx)-[:REPORTED_IN]->(pub:Publication)
    WHERE fx.workspace = $workspace
      AND ($source = 'all' OR ($source = 'local_data' AND fx.source_group = 'local_data')
           OR ($source = 'literature' AND fx.source_group = 'literature') OR $source IN ['effects','fulltext','abstract'])
      AND ($exposure_domain = 'all' OR toLower(ex.name) CONTAINS toLower($exposure_domain))
      AND ($disease_group = 'all' OR toLower(dis.name) CONTAINS toLower($disease_group))
      AND (size($terms) = 0 OR any(term IN $terms WHERE
           toLower(ex.name) CONTAINS term OR toLower(dis.name) CONTAINS term OR
           toLower(ds.name) CONTAINS term OR toLower(pub.title) CONTAINS term))
    RETURN ex.name AS exposure, dis.name AS disease, fx.measure AS measure,
           fx.estimate AS estimate, fx.ci_low AS ci_low, fx.ci_high AS ci_high,
           fx.p_value AS p_value, fx.fdr AS fdr, ds.name AS dataset,
           pub.pmid AS pmid, pub.pmcid AS pmcid, pub.doi AS doi,
           pub.title AS title, pub.source_url AS source_url, fx.snippet AS snippet,
           fx.document_id AS document_id, fx.source_group AS source_group
    LIMIT $limit
    """
    with get_driver().session(database=NEO4J_DATABASE) as session:
        return session.run(
            cypher,
            workspace=NEO4J_WORKSPACE,
            terms=terms,
            source=filters.source,
            exposure_domain=filters.exposure_domain,
            disease_group=filters.disease_group,
            limit=top_k,
        ).data()
