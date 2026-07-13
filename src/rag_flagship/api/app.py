"""The retrieval API: GET /health, POST /query, POST /ingest.

Heavy objects (the dense embedding model, the reranker, the LLM, the
Qdrant client) are built once at startup via the lifespan context
manager and stored on app.state, not rebuilt per request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx2
from fastapi import Depends, FastAPI, HTTPException, Request
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore
from llama_index.llms.ollama import Ollama
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from rag_flagship.api.schemas import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
)
from rag_flagship.chunking.factory import STRATEGY_NAMES, UnknownStrategyError, build_chunker
from rag_flagship.embeddings.dense import build_dense_embedding_model
from rag_flagship.embeddings.settings import EmbeddingSettings
from rag_flagship.generation.llm import build_generation_model
from rag_flagship.generation.pipeline import AnswerResult, answer_question
from rag_flagship.indexing.pipeline import hybrid_query, index_chunks
from rag_flagship.indexing.settings import QdrantSettings
from rag_flagship.indexing.store import build_vector_store, collection_name_for_strategy
from rag_flagship.ingestion.loader import load_processed_passages
from rag_flagship.reranking.cross_encoder import build_reranker

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@dataclass
class AppState:
    embed_model: BaseEmbedding
    reranker: CrossEncoder
    llm: Ollama
    qdrant_client: QdrantClient
    ollama_base_url: str


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    embedding_settings = EmbeddingSettings()
    qdrant_settings = QdrantSettings()
    app.state.rag = AppState(
        embed_model=build_dense_embedding_model(embedding_settings),
        reranker=build_reranker(),
        llm=build_generation_model(),
        qdrant_client=QdrantClient(url=qdrant_settings.url, api_key=qdrant_settings.api_key),
        ollama_base_url=embedding_settings.base_url,
    )
    yield
    app.state.rag.qdrant_client.close()


app = FastAPI(title="Ragulation API", lifespan=lifespan)


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.rag
    return state


def _validate_strategy(strategy: str) -> None:
    if strategy not in STRATEGY_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown strategy {strategy!r}, expected one of {STRATEGY_NAMES}",
        )


@app.get("/health")
def health(state: AppState = Depends(get_state)) -> HealthResponse:
    try:
        state.qdrant_client.get_collections()
        qdrant_reachable = True
    except Exception:
        qdrant_reachable = False

    try:
        response = httpx2.get(f"{state.ollama_base_url}/api/tags", timeout=5.0)
        ollama_reachable = response.status_code == 200
    except httpx2.HTTPError:
        ollama_reachable = False

    status = "ok" if qdrant_reachable and ollama_reachable else "degraded"
    return HealthResponse(
        status=status, qdrant_reachable=qdrant_reachable, ollama_reachable=ollama_reachable
    )


@app.post("/query")
def query(request: QueryRequest, state: AppState = Depends(get_state)) -> AnswerResult:
    _validate_strategy(request.strategy)
    vector_store = build_vector_store(
        collection_name_for_strategy(request.strategy), client=state.qdrant_client
    )

    def retriever(question: str, top_k: int) -> list[NodeWithScore]:
        return hybrid_query(vector_store, state.embed_model, question, top_k=top_k)

    return answer_question(
        request.question,
        request.language,
        retriever,
        state.reranker,  # type: ignore[arg-type]  # CrossEncoder satisfies Reranker at
        # runtime; mypy's Protocol check against its overloaded .predict fails to see it
        # (see the comment on reranking.pipeline.Reranker).
        state.llm,
        retrieval_top_k=request.retrieval_top_k,
        rerank_top_k=request.rerank_top_k,
    )


@app.post("/ingest")
def ingest(request: IngestRequest, state: AppState = Depends(get_state)) -> IngestResponse:
    _validate_strategy(request.strategy)
    try:
        chunker = build_chunker(request.strategy, state.embed_model)
    except UnknownStrategyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    passages = load_processed_passages(DEFAULT_PROCESSED_DIR)
    chunks = chunker.chunk(passages)
    if request.strategy == "parent_child" and request.leaf_only:
        chunks = [c for c in chunks if c.parent_chunk_id is not None]

    collection_name = collection_name_for_strategy(request.strategy)
    vector_store = build_vector_store(collection_name, client=state.qdrant_client)
    indexed_count = index_chunks(chunks, vector_store=vector_store, embed_model=state.embed_model)

    return IngestResponse(
        strategy=request.strategy,
        collection=collection_name,
        passages=len(passages),
        chunks_indexed=indexed_count,
    )
