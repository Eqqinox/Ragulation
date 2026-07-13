"""Extract recitals and articles from a regulation's Docling markdown export.

EUR-Lex/Cellar HTML for a regulation does not carry semantic heading tags,
so Docling exports it as flat text and one-row tables (each recital is its
own single-row table: ``| (1) | recital text |``). This module recovers
the legal structure with a small state machine over the exported markdown
instead, driven by the fixed vocabulary of EU legislative drafting
(preamble, "Whereas:", "HAVE ADOPTED THIS REGULATION:", chapters, articles,
annexes), in English and French.
"""

from __future__ import annotations

import re

from rag_flagship.corpus.manifest import CorpusDocument, Language
from rag_flagship.ingestion.models import CorpusPassage

_MARKDOWN_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_RECITAL_ROW = re.compile(r"(?m)^\|\s*\((\d+)\)\s*\|(.*)\|\s*$")

_MARKERS: dict[Language, dict[str, str]] = {
    "en": {
        "preamble_end": "Whereas:",
        "enacting_start": "HAVE ADOPTED",
        "chapter": r"^CHAPTER\s+[IVXLCDM]+\s*$",
        "section": r"^SECTION\s+\d+\s*$",
        "article": r"^Article\s+\d+\s*$",
        "annex": r"^ANNEX\b",
    },
    "fr": {
        "preamble_end": "considérant ce qui suit:",
        "enacting_start": "ONT ADOPTÉ",
        "chapter": r"^CHAPITRE\s+[IVXLCDM]+\s*$",
        "section": r"^SECTION\s+\d+\s*$",
        "article": r"^Article\s+\d+\s*$",
        "annex": r"^ANNEXE\b",
    },
}


def _strip_markdown_links(text: str) -> str:
    return _MARKDOWN_LINK.sub(r"\1", text)


def _passage(document: CorpusDocument, locator: str, text: str) -> CorpusPassage:
    return CorpusPassage(
        doc_id=document.doc_id,
        language=document.language,
        category=document.category,
        locator=locator,
        text=text,
    )


def _extract_enacting_terms(
    text: str, markers: dict[str, str], document: CorpusDocument
) -> list[CorpusPassage]:
    landmark = re.compile(
        f"{markers['chapter']}|{markers['section']}|{markers['article']}|{markers['annex']}"
    )
    passages: list[CorpusPassage] = []
    current_locator: str | None = None
    current_lines: list[str] = []
    expect_title = False

    def flush() -> None:
        if current_locator is None:
            return
        body = "\n".join(line for line in current_lines if line).strip()
        if body:
            passages.append(_passage(document, current_locator, body))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if landmark.match(line):
            flush()
            current_locator = line
            current_lines = []
            expect_title = True
            continue
        if expect_title:
            current_locator = f"{current_locator} - {line}"
            expect_title = False
            continue
        current_lines.append(line)
    flush()
    return passages


def extract_regulation_passages(markdown: str, document: CorpusDocument) -> list[CorpusPassage]:
    """Parse a regulation's Docling markdown export into cited passages.

    Produces one "Preamble" passage, one passage per numbered recital, and
    one passage per article (grouped under chapters/sections/annexes where
    present via a compound locator, for example "Article 5 - Right of
    access by the data subject").
    """
    markers = _MARKERS[document.language]
    cleaned = _strip_markdown_links(markdown)

    preamble_match = re.search(re.escape(markers["preamble_end"]), cleaned, re.IGNORECASE)
    enacting_match = re.search(re.escape(markers["enacting_start"]), cleaned, re.IGNORECASE)

    passages: list[CorpusPassage] = []

    preamble_end = preamble_match.end() if preamble_match else 0
    preamble_text = cleaned[:preamble_end].strip()
    if preamble_text:
        passages.append(_passage(document, "Preamble", preamble_text))

    enacting_start = enacting_match.start() if enacting_match else len(cleaned)
    recitals_zone = cleaned[preamble_end:enacting_start]
    for match in _RECITAL_ROW.finditer(recitals_zone):
        number, body = match.group(1), match.group(2).strip()
        if body:
            passages.append(_passage(document, f"Recital {number}", body))

    enacting_zone = cleaned[enacting_start:]
    passages.extend(_extract_enacting_terms(enacting_zone, markers, document))

    return passages
