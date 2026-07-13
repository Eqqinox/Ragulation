from rag_flagship.corpus.manifest import CorpusDocument
from rag_flagship.ingestion.regulation_parser import extract_regulation_passages

EN_DOCUMENT = CorpusDocument(
    doc_id="sample_en",
    title="Sample Regulation",
    url="https://example.invalid/sample-en",
    language="en",
    media_type="html",
    category="regulation",
    source_organization="Test",
    published="2026-01-01",
)

FR_DOCUMENT = EN_DOCUMENT.model_copy(update={"doc_id": "sample_fr", "language": "fr"})

EN_MARKDOWN = """REGULATION (EU) 2099/1 OF THE EUROPEAN PARLIAMENT AND OF THE COUNCIL

THE EUROPEAN PARLIAMENT AND THE COUNCIL OF THE EUROPEAN UNION,

Having regard to the Treaty, [( 1 )](#ntr1) ,

Whereas:

| (1)   | First recital text.   |
|-------|-----------------------|

| (2)   | Second recital text.   |
|-------|-----------------------|

HAVE ADOPTED THIS REGULATION:

CHAPTER I

General provisions

Article 1

Subject-matter

1. Body text one.
2. Body text two.

Article 2

Definitions

For the purposes of this Regulation, definitions apply.

ANNEX I

Annex heading

Annex body text.
"""

FR_MARKDOWN = """REGLEMENT (UE) 2099/1 DU PARLEMENT EUROPEEN ET DU CONSEIL

LE PARLEMENT EUROPEEN ET LE CONSEIL DE L'UNION EUROPEENNE,

vu le traite,

considérant ce qui suit:

| (1)   | Premier considerant.   |
|-------|-------------------------|

ONT ADOPTÉ LE PRÉSENT RÈGLEMENT:

CHAPITRE I

Dispositions generales

Article 1

Objet

1. Texte.
"""


def test_preamble_is_captured() -> None:
    passages = extract_regulation_passages(EN_MARKDOWN, EN_DOCUMENT)
    preamble = next(p for p in passages if p.locator == "Preamble")
    assert "Having regard to the Treaty" in preamble.text


def test_markdown_links_are_stripped_to_plain_text() -> None:
    passages = extract_regulation_passages(EN_MARKDOWN, EN_DOCUMENT)
    preamble = next(p for p in passages if p.locator == "Preamble")
    assert "[(" not in preamble.text
    assert "( 1 )" in preamble.text


def test_recitals_are_extracted_in_order() -> None:
    passages = extract_regulation_passages(EN_MARKDOWN, EN_DOCUMENT)
    recitals = [p for p in passages if p.locator.startswith("Recital")]
    assert [p.locator for p in recitals] == ["Recital 1", "Recital 2"]
    assert recitals[0].text == "First recital text."
    assert recitals[1].text == "Second recital text."


def test_articles_carry_their_title_and_body() -> None:
    passages = extract_regulation_passages(EN_MARKDOWN, EN_DOCUMENT)
    article_1 = next(p for p in passages if p.locator == "Article 1 - Subject-matter")
    assert "Body text one." in article_1.text
    assert "Body text two." in article_1.text

    article_2 = next(p for p in passages if p.locator == "Article 2 - Definitions")
    assert "definitions apply" in article_2.text


def test_annex_is_captured_not_dropped() -> None:
    passages = extract_regulation_passages(EN_MARKDOWN, EN_DOCUMENT)
    annex = next(p for p in passages if p.locator.startswith("ANNEX I"))
    assert "Annex body text." in annex.text


def test_all_passages_carry_document_metadata() -> None:
    passages = extract_regulation_passages(EN_MARKDOWN, EN_DOCUMENT)
    for passage in passages:
        assert passage.doc_id == "sample_en"
        assert passage.language == "en"
        assert passage.category == "regulation"


def test_french_markers_are_recognized() -> None:
    passages = extract_regulation_passages(FR_MARKDOWN, FR_DOCUMENT)
    locators = [p.locator for p in passages]
    assert "Recital 1" in locators
    assert "Article 1 - Objet" in locators
    recital = next(p for p in passages if p.locator == "Recital 1")
    assert recital.text == "Premier considerant."
