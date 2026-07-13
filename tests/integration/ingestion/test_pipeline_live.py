"""Runs the real Docling conversion against already-fetched corpus files.

Run explicitly with: uv run pytest -m integration
Slow (loads Docling's layout model); not run by default.
"""

from pathlib import Path

import pytest

from rag_flagship.corpus.manifest import CORPUS_MANIFEST
from rag_flagship.ingestion.pipeline import ingest_document

pytestmark = pytest.mark.integration

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


def test_gdpr_english_yields_article_1() -> None:
    document = next(doc for doc in CORPUS_MANIFEST if doc.doc_id == "gdpr_en")

    passages = ingest_document(document, raw_dir=RAW_DIR)

    locators = [p.locator for p in passages]
    assert any(locator.startswith("Article 1") for locator in locators)
    assert any(locator.startswith("Recital 1") for locator in locators)


def test_edpb_consent_guideline_yields_sections() -> None:
    document = next(doc for doc in CORPUS_MANIFEST if doc.doc_id == "edpb_consent")

    passages = ingest_document(document, raw_dir=RAW_DIR)

    assert len(passages) > 1
    assert all(p.text for p in passages)
