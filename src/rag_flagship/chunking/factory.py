"""Picks a chunking strategy instance by name.

Shared by scripts/build_index.py and the /ingest API route, so the
strategy-name-to-class mapping exists in exactly one place.
"""

from __future__ import annotations

from llama_index.core.embeddings import BaseEmbedding

from rag_flagship.chunking.base import ChunkingStrategy
from rag_flagship.chunking.parent_child import ParentChildChunker
from rag_flagship.chunking.recursive import RecursiveChunker
from rag_flagship.chunking.semantic import SemanticChunker

STRATEGY_NAMES = ("recursive", "semantic", "parent_child")


class UnknownStrategyError(ValueError):
    pass


def build_chunker(name: str, embed_model: BaseEmbedding) -> ChunkingStrategy:
    if name == "recursive":
        return RecursiveChunker()
    if name == "semantic":
        return SemanticChunker(embed_model=embed_model)
    if name == "parent_child":
        return ParentChildChunker()
    raise UnknownStrategyError(f"unknown strategy: {name!r}, expected one of {STRATEGY_NAMES}")
