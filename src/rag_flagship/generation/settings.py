"""Typed, environment-driven configuration for the generation model."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    base_url: str = "http://localhost:11434"
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
