"""Typed, environment-driven configuration for the dense embedding model.

Centralized here so the model name and server URL are set in one place,
per stateOfTheArt.md Section 6 ("configuration is centralized and typed,
no scattered os.environ reads").
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    base_url: str = "http://localhost:11434"
    dense_model_name: str = "bge-m3"
    """See ADR-0002: Ollama's bge-m3 exposes dense embeddings only."""
