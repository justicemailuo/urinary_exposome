from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .llm import vllm_status
from .graph_store import build_graph, graph_status, search_graph
from .lightrag_bridge import index_lightrag, lightrag_status
from .quota import make_client_key
from .rag import run_chat, run_search
from .schemas import ChatRequest, ChatResponse, GraphSearchResponse, IndexRequest, RagFilters, SearchResponse
from .vector_store import build_index, index_status


app = FastAPI(title="Urological Exposomics RAG v2", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "ok": True,
        "index": index_status(),
        "graph": graph_status(),
        "lightrag": lightrag_status(),
        "llm": vllm_status(),
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request) -> ChatResponse:
    forwarded_for = http_request.headers.get("x-forwarded-for", "")
    client_host = forwarded_for.split(",", 1)[0].strip() if forwarded_for else ""
    if not client_host and http_request.client:
        client_host = http_request.client.host
    user_agent = http_request.headers.get("user-agent", "")
    fallback_identity = f"{client_host}|{user_agent}"
    client_key = make_client_key(request.demo_client_id, fallback_identity)
    return await run_chat(request, client_key=client_key)


@app.post("/api/search", response_model=SearchResponse)
def search_endpoint(query: str, top_k: int = 8, filters: RagFilters | None = None) -> SearchResponse:
    return run_search(query=query, top_k=top_k, filters=filters or RagFilters())


@app.post("/api/index")
def index(reset: bool = False, batch_size: int = 64, limit: int | None = None) -> dict[str, object]:
    return build_index(batch_size=batch_size, reset=reset, limit=limit)


@app.post("/api/graph/index")
def graph_index(request: IndexRequest) -> dict[str, object]:
    return build_graph(reset=request.reset, batch_size=request.batch_size, limit=request.limit)


@app.post("/api/graph/search", response_model=GraphSearchResponse)
def graph_search(query: str, top_k: int = 8, filters: RagFilters | None = None) -> GraphSearchResponse:
    import time
    started = time.time()
    paths = search_graph(query, top_k, filters or RagFilters())
    return GraphSearchResponse(paths=paths, elapsed_ms=int((time.time() - started) * 1000))


@app.post("/api/lightrag/index")
async def lightrag_index(request: IndexRequest) -> dict[str, object]:
    if request.reset:
        return {"error": "LightRAG reset is intentionally not exposed; use a new NEO4J_WORKSPACE."}
    return await index_lightrag(batch_size=min(request.batch_size, 100), limit=request.limit)
