"""Typed, environment-driven configuration for the cross-encoder reranker."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER_", extra="ignore")

    model_name: str = "BAAI/bge-reranker-v2-m3"
    """See ADR-0004: Apache-2.0, multilingual, same BAAI/BGE family as bge-m3."""
    device: str | None = None
    """None lets sentence-transformers auto-detect (cuda, mps, or cpu)."""
