"""Typed, environment-driven configuration for the cross-encoder reranker."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class RerankerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RERANKER_", extra="ignore")

    model_name: str = "BAAI/bge-reranker-v2-m3"
    """See ADR-0004: Apache-2.0, multilingual, same BAAI/BGE family as bge-m3."""
    device: str = "cpu"
    """Deliberately not auto-detected. On this machine, letting
    sentence-transformers pick MPS (Apple's GPU backend) causes silent,
    wrong results (identical scores for genuinely different passages,
    with an Insufficient Memory Metal error in stderr) when Ollama's
    models are also using the GPU. CPU inference for this ~500 MB model
    is deterministic and fast enough; see ADR-0004's addendum."""
