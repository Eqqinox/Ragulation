from dataclasses import dataclass

from rag_flagship.corpus.manifest import CorpusDocument
from rag_flagship.ingestion.guidance_parser import extract_guidance_passages

DOCUMENT = CorpusDocument(
    doc_id="sample_guideline",
    title="Sample Guideline",
    url="https://example.invalid/sample-guideline",
    language="en",
    media_type="pdf",
    category="guideline",
    source_organization="Test",
    published="2026-01-01",
)


@dataclass
class FakeItem:
    label: str
    text: str


def test_text_before_first_heading_is_kept_as_front_matter() -> None:
    items = [FakeItem("text", "Guidelines 1/2026"), FakeItem("text", "Adopted on 1 January 2026")]

    passages = extract_guidance_passages(items, DOCUMENT)

    assert len(passages) == 1
    assert passages[0].locator == "Front matter"
    assert "Guidelines 1/2026" in passages[0].text


def test_sections_are_grouped_by_heading() -> None:
    items = [
        FakeItem("section_header", "1 INTRODUCTION"),
        FakeItem("text", "This is the introduction."),
        FakeItem("list_item", "A list point."),
        FakeItem("section_header", "2 SCOPE"),
        FakeItem("text", "This is the scope."),
    ]

    passages = extract_guidance_passages(items, DOCUMENT)

    assert [p.locator for p in passages] == ["1 INTRODUCTION", "2 SCOPE"]
    assert "This is the introduction." in passages[0].text
    assert "A list point." in passages[0].text
    assert "This is the scope." in passages[1].text


def test_page_headers_and_footers_are_ignored() -> None:
    items = [
        FakeItem("section_header", "1 INTRODUCTION"),
        FakeItem("page_footer", "adopted"),
        FakeItem("page_footer", "3"),
        FakeItem("text", "Real content."),
    ]

    passages = extract_guidance_passages(items, DOCUMENT)

    assert len(passages) == 1
    assert passages[0].text == "Real content."


def test_empty_sections_are_dropped() -> None:
    items = [
        FakeItem("section_header", "1 EMPTY"),
        FakeItem("section_header", "2 REAL"),
        FakeItem("text", "Content."),
    ]

    passages = extract_guidance_passages(items, DOCUMENT)

    assert [p.locator for p in passages] == ["2 REAL"]


def test_passages_carry_document_metadata() -> None:
    items = [FakeItem("section_header", "1 INTRO"), FakeItem("text", "Body.")]

    passages = extract_guidance_passages(items, DOCUMENT)

    assert passages[0].doc_id == "sample_guideline"
    assert passages[0].language == "en"
    assert passages[0].category == "guideline"
