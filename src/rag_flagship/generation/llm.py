"""Factory for the production generation model."""

from __future__ import annotations

from llama_index.llms.ollama import Ollama

from rag_flagship.generation.settings import GenerationSettings


def build_generation_model(settings: GenerationSettings | None = None) -> Ollama:
    active_settings = settings if settings is not None else GenerationSettings()
    headers = (
        {"Authorization": f"Bearer {active_settings.generation_api_key}"}
        if active_settings.generation_api_key
        else None
    )
    model: Ollama = Ollama(
        model=active_settings.generation_model_name,
        base_url=active_settings.generation_base_url,
        request_timeout=active_settings.request_timeout_seconds,
        context_window=active_settings.context_window,
        headers=headers,
    )
    return model
