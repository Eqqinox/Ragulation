"""Effectful wrapper around Docling's DocumentConverter.

Kept separate from the pure parsers in ``regulation_parser.py`` and
``guidance_parser.py`` so unit tests for parsing logic never need to load
Docling's ML models; only the integration tests exercise this module.
"""

from __future__ import annotations

from pathlib import Path

from docling.document_converter import DocumentConverter

from rag_flagship.ingestion.guidance_parser import DoclingTextItem


def convert_to_markdown(path: Path, converter: DocumentConverter | None = None) -> str:
    """Convert a regulation HTML file to its Docling markdown export."""
    active_converter = converter if converter is not None else DocumentConverter()
    result = active_converter.convert(str(path))
    return result.document.export_to_markdown()


def convert_to_items(
    path: Path, converter: DocumentConverter | None = None
) -> list[DoclingTextItem]:
    """Convert a guidance PDF and return its text items in reading order."""
    active_converter = converter if converter is not None else DocumentConverter()
    result = active_converter.convert(str(path))
    return list(result.document.texts)
