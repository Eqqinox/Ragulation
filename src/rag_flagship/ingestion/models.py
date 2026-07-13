"""The output type of ingestion: one citable passage of a source document."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rag_flagship.corpus.manifest import Category, Language


class CorpusPassage(BaseModel):
    """One structural unit of a parsed document (a recital, an article, a
    guidance section, and so on), with enough metadata to cite it."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    language: Language
    category: Category
    locator: str
    """Human-readable reference for citation, for example "Article 5" or
    "Recital 12" or a guidance document's own section heading."""
    text: str
