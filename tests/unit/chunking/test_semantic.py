import hashlib
import math
import re
from collections.abc import Sequence

from llama_index.core.embeddings import BaseEmbedding

from rag_flagship.chunking.semantic import SemanticChunker
from rag_flagship.corpus.manifest import CorpusDocument
from rag_flagship.ingestion.models import CorpusPassage

DOCUMENT = CorpusDocument(
    doc_id="sample_doc",
    title="Sample",
    url="https://example.invalid/sample",
    language="en",
    media_type="html",
    category="regulation",
    source_organization="Test",
    published="2026-01-01",
)


def _passage(text: str, locator: str = "Article 1") -> CorpusPassage:
    return CorpusPassage(
        doc_id=DOCUMENT.doc_id,
        language=DOCUMENT.language,
        category=DOCUMENT.category,
        locator=locator,
        text=text,
    )


def _hash_embedding(text: str, dims: int = 32) -> list[float]:
    """Deterministic bag-of-words embedding: texts sharing vocabulary land
    close together, texts with different vocabulary land far apart. Good
    enough to exercise SemanticSplitterNodeParser's breakpoint logic
    without a real model."""
    vector = [0.0] * dims
    for word in re.findall(r"\w+", text.lower()):
        digest = hashlib.md5(word.encode(), usedforsecurity=False).hexdigest()
        bucket = int(digest, 16) % dims
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeEmbedding(BaseEmbedding):
    def _get_text_embedding(self, text: str) -> list[float]:
        return _hash_embedding(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return _hash_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return _hash_embedding(query)

    def _get_text_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        return [_hash_embedding(text) for text in texts]


def test_produces_at_least_one_chunk_per_passage() -> None:
    chunker = SemanticChunker(embed_model=FakeEmbedding())
    text = "Cats are small animals. Cats like to sleep. Cats hunt mice. " * 3

    chunks = chunker.chunk([_passage(text)])

    assert len(chunks) >= 1


def test_distinct_topics_split_into_different_chunks() -> None:
    chunker = SemanticChunker(embed_model=FakeEmbedding(), breakpoint_percentile_threshold=50)
    topic_one = "Cats are small animals. Cats like to sleep. Cats hunt mice. " * 5
    topic_two = (
        "Regulation compliance requires audits. Regulation compliance requires "
        "documentation. Regulation compliance requires monitoring. "
    ) * 5

    chunks = chunker.chunk([_passage(topic_one + topic_two)])

    assert len(chunks) >= 2


def test_every_chunk_is_a_substring_of_the_source_passage() -> None:
    chunker = SemanticChunker(embed_model=FakeEmbedding())
    text = "First idea here. Second idea follows. Third idea concludes. " * 3

    chunks = chunker.chunk([_passage(text)])

    for chunk in chunks:
        assert chunk.text in text


def test_all_chunks_carry_strategy_and_metadata() -> None:
    chunker = SemanticChunker(embed_model=FakeEmbedding())
    chunks = chunker.chunk([_passage("Some content here. " * 10, locator="Recital 9")])

    for chunk in chunks:
        assert chunk.strategy == "semantic"
        assert chunk.locator == "Recital 9"
        assert chunk.doc_id == "sample_doc"
