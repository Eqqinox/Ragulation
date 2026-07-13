"""Factory for the production cross-encoder reranker."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_flagship.reranking.settings import RerankerSettings


def build_reranker(settings: RerankerSettings | None = None) -> CrossEncoder:
    active_settings = settings if settings is not None else RerankerSettings()
    model: CrossEncoder = CrossEncoder(active_settings.model_name, device=active_settings.device)
    return model
