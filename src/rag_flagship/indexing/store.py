"""Factory for the Qdrant hybrid (dense + BM25 sparse) vector store.

See ADR-0002: dense vectors come from Ollama bge-m3 (this module does not
compute them, callers pass an embed_model to pipeline.index_chunks), and
the sparse side is BM25 via fastembed's Qdrant/bm25 model, computed by
Qdrant's own client library, not by this codebase.
"""

from __future__ import annotations

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from rag_flagship.indexing.settings import QdrantSettings

SPARSE_MODEL = "Qdrant/bm25"


def collection_name_for_strategy(strategy: str) -> str:
    return f"rag_flagship_{strategy}"


def build_vector_store(
    collection_name: str,
    settings: QdrantSettings | None = None,
    client: QdrantClient | None = None,
) -> QdrantVectorStore:
    active_settings = settings if settings is not None else QdrantSettings()
    active_client = (
        client
        if client is not None
        else QdrantClient(url=active_settings.url, api_key=active_settings.api_key)
    )
    return QdrantVectorStore(
        collection_name=collection_name,
        client=active_client,
        enable_hybrid=True,
        fastembed_sparse_model=SPARSE_MODEL,
    )
