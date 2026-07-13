"""Group a guidance PDF's Docling text items into cited sections.

Unlike the regulation HTML (see ``regulation_parser.py``), Docling's PDF
layout model reliably detects section headings as their own labeled items
for these EDPB/Commission guidance documents, so no bespoke pattern
matching is needed: passages are simply the text between one
``section_header`` item and the next.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from rag_flagship.corpus.manifest import CorpusDocument
from rag_flagship.ingestion.models import CorpusPassage

BODY_LABELS = frozenset({"text", "list_item", "footnote"})
SECTION_HEADER_LABEL = "section_header"
IGNORED_LABELS = frozenset({"page_header", "page_footer"})


class DoclingTextItem(Protocol):
    """Structural typing helper: anything with ``.label`` and ``.text``.

    Declared as read-only properties, not plain attributes, so it matches
    covariantly: Docling's real item classes type ``label`` as the
    ``DocItemLabel`` enum (a ``str`` subclass), not literally ``str``.
    """

    @property
    def label(self) -> str: ...
    @property
    def text(self) -> str: ...


def extract_guidance_passages(
    items: Iterable[DoclingTextItem], document: CorpusDocument
) -> list[CorpusPassage]:
    """Group items into one passage per section heading.

    Text preceding the first heading (title, version history, table of
    contents boilerplate) is kept under the locator "Front matter" rather
    than dropped, so nothing is silently lost.
    """
    passages: list[CorpusPassage] = []
    current_locator = "Front matter"
    current_lines: list[str] = []

    def flush() -> None:
        body = "\n".join(line for line in current_lines if line).strip()
        if body:
            passages.append(
                CorpusPassage(
                    doc_id=document.doc_id,
                    language=document.language,
                    category=document.category,
                    locator=current_locator,
                    text=body,
                )
            )

    for item in items:
        label = item.label
        if label in IGNORED_LABELS:
            continue
        if label == SECTION_HEADER_LABEL:
            flush()
            current_locator = item.text.strip() or current_locator
            current_lines = []
            continue
        if label in BODY_LABELS:
            text = item.text.strip()
            if text:
                current_lines.append(text)
    flush()
    return passages
