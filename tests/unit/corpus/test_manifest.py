from rag_flagship.corpus.manifest import CORPUS_MANIFEST


def test_document_ids_are_unique() -> None:
    ids = [doc.doc_id for doc in CORPUS_MANIFEST]
    assert len(ids) == len(set(ids))


def test_core_regulations_have_english_and_french_texts() -> None:
    for celex in ("32016R0679", "32024R1689"):
        languages = {
            doc.language
            for doc in CORPUS_MANIFEST
            if doc.category == "regulation" and doc.celex == celex
        }
        assert languages == {"en", "fr"}


def test_all_urls_use_https() -> None:
    for doc in CORPUS_MANIFEST:
        assert str(doc.url).startswith("https://")


def test_guideline_and_code_of_practice_documents_are_english_only() -> None:
    for doc in CORPUS_MANIFEST:
        if doc.category in ("guideline", "code_of_practice"):
            assert doc.language == "en"


def test_regulation_accept_language_header_matches_document_language() -> None:
    expected = {"en": "eng", "fr": "fra"}
    for doc in CORPUS_MANIFEST:
        if doc.category == "regulation":
            assert doc.request_headers.get("Accept-Language") == expected[doc.language]
