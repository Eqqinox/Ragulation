import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rag_flagship.chunking.parent_child import ParentChildChunker
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


_sentence_words = st.text(alphabet=string.ascii_letters, min_size=1, max_size=12)
_sentences = st.lists(_sentence_words, min_size=1, max_size=80).map(
    lambda words: ". ".join(words) + "."
)


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(text=_sentences)
def test_every_child_parent_id_points_to_a_real_chunk(text: str) -> None:
    chunker = ParentChildChunker(chunk_sizes=[64, 16], chunk_overlap=4)

    chunks = chunker.chunk([_passage(text)])

    chunk_ids = {c.chunk_id for c in chunks}
    for chunk in chunks:
        if chunk.parent_chunk_id is not None:
            assert chunk.parent_chunk_id in chunk_ids


@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(text=_sentences)
def test_every_chunk_is_a_substring_of_the_source_passage(text: str) -> None:
    chunker = ParentChildChunker(chunk_sizes=[64, 16], chunk_overlap=4)

    chunks = chunker.chunk([_passage(text)])

    for chunk in chunks:
        assert chunk.text in text


def test_at_least_one_top_level_chunk_has_no_parent() -> None:
    chunker = ParentChildChunker(chunk_sizes=[64, 16], chunk_overlap=4)
    chunks = chunker.chunk([_passage("Some content here. " * 40)])

    assert any(chunk.parent_chunk_id is None for chunk in chunks)


def test_all_chunks_carry_strategy_and_metadata() -> None:
    chunker = ParentChildChunker(chunk_sizes=[64, 16], chunk_overlap=4)
    chunks = chunker.chunk([_passage("Some content here. " * 40, locator="Recital 3")])

    for chunk in chunks:
        assert chunk.strategy == "parent_child"
        assert chunk.locator == "Recital 3"
        assert chunk.doc_id == "sample_doc"
