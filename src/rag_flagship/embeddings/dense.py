"""Factory for the production dense embedding model.

The single place that knows the embedding model is Ollama's bge-m3;
nothing else in the codebase should construct an OllamaEmbedding or
hardcode the model name directly.
"""

from __future__ import annotations

from llama_index.embeddings.ollama import OllamaEmbedding

from rag_flagship.embeddings.settings import EmbeddingSettings


def build_dense_embedding_model(settings: EmbeddingSettings | None = None) -> OllamaEmbedding:
    active_settings = settings if settings is not None else EmbeddingSettings()
    return OllamaEmbedding(
        model_name=active_settings.dense_model_name,
        base_url=active_settings.base_url,
    )
