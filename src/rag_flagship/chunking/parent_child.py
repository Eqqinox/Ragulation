"""Parent-child (hierarchical) chunking.

Wraps LlamaIndex's HierarchicalNodeParser: splits each passage into large
parent blocks, then splits each parent into smaller child chunks. Both
levels are returned; retrieval (a later stage) indexes only the leaf
chunks (``parent_chunk_id is not None``) and can expand a hit back to its
parent's full text for generation context ("auto-merging" retrieval).
"""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import NodeRelationship

from rag_flagship.chunking.models import Chunk
from rag_flagship.ingestion.models import CorpusPassage

DEFAULT_CHUNK_SIZES = [1024, 256]
DEFAULT_CHUNK_OVERLAP = 20


class ParentChildChunker:
    def __init__(
        self,
        chunk_sizes: list[int] | None = None,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=chunk_sizes or list(DEFAULT_CHUNK_SIZES),
            chunk_overlap=chunk_overlap,
        )

    def chunk(self, passages: Sequence[CorpusPassage]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for passage in passages:
            document = Document(text=passage.text)
            nodes = self._parser.get_nodes_from_documents([document])

            chunk_id_by_node_id = {
                node.node_id: f"{passage.doc_id}:{passage.locator}:parent_child:{index}"
                for index, node in enumerate(nodes)
            }

            for node in nodes:
                parent_relationship = node.relationships.get(NodeRelationship.PARENT)
                if isinstance(parent_relationship, list):
                    parent_relationship = parent_relationship[0] if parent_relationship else None
                parent_chunk_id = (
                    chunk_id_by_node_id.get(parent_relationship.node_id)
                    if parent_relationship
                    else None
                )
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id_by_node_id[node.node_id],
                        doc_id=passage.doc_id,
                        language=passage.language,
                        category=passage.category,
                        locator=passage.locator,
                        strategy="parent_child",
                        text=node.get_content(),
                        parent_chunk_id=parent_chunk_id,
                    )
                )
        return chunks
