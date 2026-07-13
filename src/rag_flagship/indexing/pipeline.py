"""Turns Chunk records into embedded, indexed Qdrant points.

Qdrant point IDs must be an unsigned integer or a UUID (a plain
chunk_id string like "gdpr_en:Article 5:recursive:0" is rejected by the
server), so each chunk_id is mapped to a UUID5 derived from it, keeping
the point ID stable and idempotent across re-indexing runs. The original
chunk_id is preserved in the node's payload for citation.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    VectorStoreQueryMode,
)

from rag_flagship.chunking.models import Chunk

CHUNK_ID_NAMESPACE = uuid.NAMESPACE_URL


def chunk_to_node(chunk: Chunk) -> TextNode:
    point_id = str(uuid.uuid5(CHUNK_ID_NAMESPACE, chunk.chunk_id))
    return TextNode(
        id_=point_id,
        text=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "language": chunk.language,
            "category": chunk.category,
            "locator": chunk.locator,
            "strategy": chunk.strategy,
            "parent_chunk_id": chunk.parent_chunk_id,
        },
    )


def index_chunks(
    chunks: Sequence[Chunk],
    vector_store: BasePydanticVectorStore,
    embed_model: BaseEmbedding,
) -> int:
    """Embed and upsert every chunk. Returns the number of chunks indexed."""
    if not chunks:
        return 0
    nodes = [chunk_to_node(chunk) for chunk in chunks]
    texts = [node.get_content() for node in nodes]
    embeddings = embed_model.get_text_embedding_batch(texts)
    for node, embedding in zip(nodes, embeddings, strict=True):
        node.embedding = embedding

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    VectorStoreIndex(nodes=nodes, storage_context=storage_context, embed_model=embed_model)
    return len(nodes)


def hybrid_query(
    vector_store: BasePydanticVectorStore,
    embed_model: BaseEmbedding,
    query_text: str,
    top_k: int = 5,
) -> list[NodeWithScore]:
    """Run a dense + BM25 sparse query, fused with Qdrant's RRF, returning
    the top_k ranked results."""
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = index.as_retriever(
        vector_store_query_mode=VectorStoreQueryMode.HYBRID,
        similarity_top_k=top_k,
    )
    return retriever.retrieve(query_text)
