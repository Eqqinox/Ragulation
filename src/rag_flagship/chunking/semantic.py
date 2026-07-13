"""Semantic (embedding-breakpoint) chunking.

Wraps LlamaIndex's SemanticSplitterNodeParser: splits into sentences, then
groups consecutive sentences together as long as each new sentence's
embedding stays similar to the running group, cutting a new chunk where
cosine dissimilarity crosses the breakpoint percentile. Requires an
embedding model; production use injects the Ollama bge-m3 client from
``rag_flagship.embeddings`` (see ADR-0002), unit tests inject a fake.
"""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SemanticSplitterNodeParser

from rag_flagship.chunking.models import Chunk
from rag_flagship.ingestion.models import CorpusPassage

DEFAULT_BUFFER_SIZE = 1
DEFAULT_BREAKPOINT_PERCENTILE = 95


class SemanticChunker:
    def __init__(
        self,
        embed_model: BaseEmbedding,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        breakpoint_percentile_threshold: int = DEFAULT_BREAKPOINT_PERCENTILE,
    ) -> None:
        self._splitter = SemanticSplitterNodeParser(
            embed_model=embed_model,
            buffer_size=buffer_size,
            breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        )

    def chunk(self, passages: Sequence[CorpusPassage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for passage in passages:
            document = Document(text=passage.text)
            nodes = self._splitter.get_nodes_from_documents([document])
            for index, node in enumerate(nodes):
                chunks.append(
                    Chunk(
                        chunk_id=f"{passage.doc_id}:{passage.locator}:semantic:{index}",
                        doc_id=passage.doc_id,
                        language=passage.language,
                        category=passage.category,
                        locator=passage.locator,
                        strategy="semantic",
                        text=node.get_content(),
                    )
                )
        return chunks
