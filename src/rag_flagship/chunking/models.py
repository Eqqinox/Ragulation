"""The output type of chunking: one retrieval-sized unit of a passage."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from rag_flagship.corpus.manifest import Category, Language

Strategy = Literal["recursive", "semantic", "parent_child"]


class Chunk(BaseModel):
    """One chunk produced by a chunking strategy from a single passage."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    language: Language
    category: Category
    locator: str
    """Inherited from the source passage; several chunks share one locator."""
    strategy: Strategy
    text: str
    parent_chunk_id: str | None = None
    """Set only by the parent_child strategy: the chunk_id of the larger
    context block this chunk was split from. None for top-level chunks."""
