"""Request/response schemas for the retrieval API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_flagship.chunking.factory import STRATEGY_NAMES
from rag_flagship.corpus.manifest import Language
from rag_flagship.generation.pipeline import DEFAULT_RERANK_TOP_K, DEFAULT_RETRIEVAL_TOP_K


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    language: Language = "en"
    strategy: str = Field(default="recursive", description=f"One of {STRATEGY_NAMES}")
    retrieval_top_k: int = Field(default=DEFAULT_RETRIEVAL_TOP_K, ge=1, le=100)
    rerank_top_k: int = Field(default=DEFAULT_RERANK_TOP_K, ge=1, le=50)


class IngestRequest(BaseModel):
    strategy: str = Field(default="recursive", description=f"One of {STRATEGY_NAMES}")
    leaf_only: bool = True


class IngestResponse(BaseModel):
    strategy: str
    collection: str
    passages: int
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    qdrant_reachable: bool
    ollama_reachable: bool
