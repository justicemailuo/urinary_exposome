from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


PROJECT_DIR = Path(os.environ.get("URINARY_RAG_PROJECT_DIR", Path(__file__).resolve().parents[2])).resolve()
DATA_DIR = PROJECT_DIR / "data"
RAG_DIR = DATA_DIR / "rag"
EFFECT_DIR = DATA_DIR / "effects"
VECTOR_DIR = PROJECT_DIR / "data" / "chroma_bge_m3"
LIGHTRAG_DIR = PROJECT_DIR / "data" / "lightrag_storage"

EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
CHROMA_COLLECTION = os.environ.get("RAG_CHROMA_COLLECTION", "urinary_exposome_bge_m3")

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8001/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "EMPTY")
LLM_TIMEOUT_SECONDS = int(os.environ.get("RAG_LLM_TIMEOUT", "120"))

DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "8"))
MAX_CONTEXT_CHARS = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "2600"))

GRAPH_ENABLED = os.environ.get("RAG_GRAPH_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")
NEO4J_WORKSPACE = os.environ.get("NEO4J_WORKSPACE", "urological_expomics")

LIGHTRAG_ENABLED = os.environ.get("RAG_LIGHTRAG_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
LIGHTRAG_MODE = os.environ.get("RAG_LIGHTRAG_MODE", "hybrid")
LIGHTRAG_TOP_K = int(os.environ.get("RAG_LIGHTRAG_TOP_K", "20"))
