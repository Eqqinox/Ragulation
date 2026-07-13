"""The common interface every chunking strategy implements."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_flagship.chunking.models import Chunk
from rag_flagship.ingestion.models import CorpusPassage


class ChunkingStrategy(Protocol):
    def chunk(self, passages: Sequence[CorpusPassage]) -> list[Chunk]: ...
