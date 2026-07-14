"""Runs the real Qdrant hybrid indexing and retrieval pipeline.

Run explicitly with: uv run pytest -m integration
Requires a local Qdrant instance (docker compose up -d) and a local
Ollama server with bge-m3 pulled (ollama pull bge-m3).
"""

import pytest
from qdrant_client import QdrantClient

from rag_flagship.chunking.models import Chunk
from rag_flagship.embeddings.dense import build_dense_embedding_model
from rag_flagship.indexing.pipeline import dense_query, hybrid_query, index_chunks
from rag_flagship.indexing.settings import QdrantSettings
from rag_flagship.indexing.store import build_vector_store

pytestmark = pytest.mark.integration

TEST_COLLECTION = "rag_flagship_test_integration"

CHUNKS = [
    Chunk(
        chunk_id="gdpr_en:Article 15:recursive:0",
        doc_id="gdpr_en",
        language="en",
        category="regulation",
        locator="Article 15",
        strategy="recursive",
        text="The data subject has the right to access their personal data.",
    ),
    Chunk(
        chunk_id="ai_act_en:Article 53:recursive:0",
        doc_id="ai_act_en",
        language="en",
        category="regulation",
        locator="Article 53",
        strategy="recursive",
        text="Providers of general-purpose AI models shall keep technical documentation.",
    ),
]


@pytest.fixture
def clean_collection():
    settings = QdrantSettings()
    client = QdrantClient(url=settings.url, api_key=settings.api_key)
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)
    yield client
    if client.collection_exists(TEST_COLLECTION):
        client.delete_collection(TEST_COLLECTION)


def test_indexed_chunks_are_retrievable_by_hybrid_query(clean_collection) -> None:
    embed_model = build_dense_embedding_model()
    vector_store = build_vector_store(TEST_COLLECTION, client=clean_collection)

    indexed_count = index_chunks(CHUNKS, vector_store=vector_store, embed_model=embed_model)
    assert indexed_count == len(CHUNKS)

    results = hybrid_query(
        vector_store,
        embed_model,
        "What documentation must AI model providers maintain?",
        top_k=2,
    )

    assert len(results) >= 1
    assert results[0].node.metadata["locator"] == "Article 53"


def test_indexed_chunks_are_retrievable_by_dense_query(clean_collection) -> None:
    embed_model = build_dense_embedding_model()
    vector_store = build_vector_store(TEST_COLLECTION, client=clean_collection)

    indexed_count = index_chunks(CHUNKS, vector_store=vector_store, embed_model=embed_model)
    assert indexed_count == len(CHUNKS)

    results = dense_query(
        vector_store,
        embed_model,
        "What documentation must AI model providers maintain?",
        top_k=2,
    )

    assert len(results) >= 1
    assert results[0].node.metadata["locator"] == "Article 53"
