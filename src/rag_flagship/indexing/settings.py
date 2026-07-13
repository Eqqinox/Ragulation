"""Typed, environment-driven configuration for the Qdrant vector store."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", extra="ignore")

    url: str = "http://localhost:6333"
    api_key: str | None = None
