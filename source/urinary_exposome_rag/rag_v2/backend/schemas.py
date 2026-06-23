from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class RagFilters(BaseModel):
    source: Literal["all", "local_data", "literature", "effects", "fulltext", "abstract"] = "all"
    exposure_domain: str = "all"
    disease_group: str = "all"
    effects_only: bool = False
    table_only: bool = False
    chinese_only: bool = False


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(8, ge=1, le=30)
    filters: RagFilters = Field(default_factory=RagFilters)
    use_llm: bool = True
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    use_graph: bool = True
    use_lightrag: bool = False
    demo_client_id: str | None = Field(default=None, max_length=128)
    user_api_key: str | None = Field(default=None, max_length=4096)
    user_base_url: str | None = Field(default=None, max_length=512)
    user_model: str | None = Field(default=None, max_length=256)


class Source(BaseModel):
    rank: int
    score: float
    id: str
    title: str = ""
    text: str = ""
    source_group: Literal["local_data", "literature"]
    source_group_label: str
    collection: str = ""
    source_type: str = ""
    source_label: str = ""
    source_url: str = ""
    pmid: str = ""
    pmcid: str = ""
    doi: str = ""
    exposure_domains: list[str] = Field(default_factory=list)
    disease_groups: list[str] = Field(default_factory=list)
    effect: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    answer: str
    retrieval_answer: str | None = None
    sources: list[Source]
    source_groups: dict[str, int]
    llm_used: bool = False
    llm_error: str | None = None
    graph_paths: list[dict[str, Any]] = Field(default_factory=list)
    graph_error: str | None = None
    lightrag_context: str | None = None
    lightrag_error: str | None = None
    llm_provider: str | None = None
    demo_usage_used: int | None = None
    demo_usage_remaining: int | None = None
    demo_usage_limit: int | None = None
    needs_user_api_key: bool = False
    elapsed_ms: int


class SearchResponse(BaseModel):
    sources: list[Source]
    source_groups: dict[str, int]
    elapsed_ms: int


class GraphSearchResponse(BaseModel):
    paths: list[dict[str, Any]] = Field(default_factory=list)
    elapsed_ms: int


class IndexRequest(BaseModel):
    reset: bool = False
    batch_size: int = Field(250, ge=1, le=5000)
    limit: int | None = Field(None, ge=1)
