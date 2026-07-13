"""Recursive (sentence-boundary-aware, fixed-size) chunking.

Wraps LlamaIndex's SentenceSplitter: splits on paragraph, then sentence,
then word boundaries, recursively, until each chunk fits chunk_size
tokens, keeping chunk_overlap tokens of context between consecutive
chunks. This is the baseline strategy the other two are compared against.
"""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from rag_flagship.chunking.models import Chunk
from rag_flagship.ingestion.models import CorpusPassage

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64


class RecursiveChunker:
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def chunk(self, passages: Sequence[CorpusPassage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for passage in passages:
            document = Document(text=passage.text)
            nodes = self._splitter.get_nodes_from_documents([document])
            for index, node in enumerate(nodes):
                chunks.append(
                    Chunk(
                        chunk_id=f"{passage.doc_id}:{passage.locator}:recursive:{index}",
                        doc_id=passage.doc_id,
                        language=passage.language,
                        category=passage.category,
                        locator=passage.locator,
                        strategy="recursive",
                        text=node.get_content(),
                    )
                )
        return chunks
