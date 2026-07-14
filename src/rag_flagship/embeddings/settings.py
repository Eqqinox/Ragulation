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
    api_key: str = ""
    """Empty for a local Ollama server (no auth). Set to an Ollama Cloud
    API key, with base_url=https://ollama.com, to run embeddings there
    instead -- used by the CI faithfulness gate, since GitHub-hosted
    runners are CPU-only and were measurably too slow embedding even a
    single document locally (~14 minutes). See ADR-0006."""
