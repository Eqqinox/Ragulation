import string

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rag_flagship.chunking.recursive import RecursiveChunker
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
_sentences = st.lists(_sentence_words, min_size=1, max_size=40).map(
    lambda words: ". ".join(words) + "."
)


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(text=_sentences)
def test_every_chunk_is_a_substring_of_the_source_passage(text: str) -> None:
    chunker = RecursiveChunker(chunk_size=32, chunk_overlap=4)

    chunks = chunker.chunk([_passage(text)])

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.text in text


@settings(max_examples=25, suppress_health_check=[HealthCheck.too_slow], deadline=None)
@given(text=_sentences)
def test_every_chunk_carries_source_passage_metadata(text: str) -> None:
    chunker = RecursiveChunker(chunk_size=32, chunk_overlap=4)

    chunks = chunker.chunk([_passage(text, locator="Article 7")])

    for chunk in chunks:
        assert chunk.doc_id == "sample_doc"
        assert chunk.language == "en"
        assert chunk.category == "regulation"
        assert chunk.locator == "Article 7"
        assert chunk.strategy == "recursive"


def test_multiple_passages_are_chunked_independently() -> None:
    chunker = RecursiveChunker(chunk_size=32, chunk_overlap=4)
    passages = [
        _passage("First passage content. " * 10, locator="Recital 1"),
        _passage("Second passage content. " * 10, locator="Recital 2"),
    ]

    chunks = chunker.chunk(passages)

    locators = {chunk.locator for chunk in chunks}
    assert locators == {"Recital 1", "Recital 2"}


def test_chunk_ids_are_unique() -> None:
    chunker = RecursiveChunker(chunk_size=16, chunk_overlap=2)
    chunks = chunker.chunk([_passage("Some text here. " * 20)])

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
