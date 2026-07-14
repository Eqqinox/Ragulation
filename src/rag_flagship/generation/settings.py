"""Typed, environment-driven configuration for the generation model."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    generation_base_url: str = "http://localhost:11434"
    """Deliberately its own field, not shared with EmbeddingSettings'
    base_url: the CI faithfulness gate points generation at Ollama Cloud
    while keeping embeddings local (Ollama Cloud's catalog is large
    frontier models only, no embedding models -- see ADR-0006's
    addendum), so the two must be independently overridable via
    distinct env vars in the same job."""
    generation_model_name: str = "mistral-small3.2"
    """See ADR-0005: already pulled locally, matches the brief's Option B."""
    request_timeout_seconds: float = 300.0
    """mistral-small3.2 is a 15 GB model; a cold call with several
    retrieved passages in context measurably exceeded 120s in local
    testing on this machine."""
    context_window: int = 8192
    """Explicit, not left at llama-index-llms-ollama's default (which
    requests the model's full context, 131072 tokens for mistral-small3.2).
    On this machine (24 GB unified memory), that default crashed the
    Ollama server outright (unexpected EOF, status 500) on a real call;
    8192 tokens is far more than this pipeline's prompt (a system
    instruction plus 5 retrieved passages) ever needs."""
    generation_api_key: str = ""
    """Empty for a local Ollama server (no auth). Set alongside
    generation_base_url=https://ollama.com to generate via Ollama Cloud
    instead -- used by the CI faithfulness gate. See ADR-0006."""
