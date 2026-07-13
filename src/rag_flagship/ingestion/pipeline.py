"""Ties the Docling adapter to the per-category pure parser."""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter

from rag_flagship.corpus.fetcher import raw_file_path
from rag_flagship.corpus.manifest import CorpusDocument
from rag_flagship.ingestion.converter import convert_to_items, convert_to_markdown
from rag_flagship.ingestion.guidance_parser import extract_guidance_passages
from rag_flagship.ingestion.models import CorpusPassage
from rag_flagship.ingestion.regulation_parser import extract_regulation_passages


def ingest_document(
    document: CorpusDocument,
    raw_dir: Path,
    converter: DocumentConverter | None = None,
) -> list[CorpusPassage]:
    """Parse one already-fetched raw source file into cited passages."""
    path = raw_file_path(document, raw_dir)
    if document.category == "regulation":
        markdown = convert_to_markdown(path, converter=converter)
        return extract_regulation_passages(markdown, document)
    items = convert_to_items(path, converter=converter)
    return extract_guidance_passages(items, document)
