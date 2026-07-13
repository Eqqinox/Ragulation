"""The schema for one hand-curated question/answer pair in the golden set.

Consumed by the RAGAS evaluation pipeline (Semaine 3), not by anything in
Semaine 1; this module exists now so the schema is validated (this file)
and enforced (data/golden/qa_v1.jsonl) from day one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from rag_flagship.corpus.manifest import Language

GoldenCategory = Literal["factual", "multi_hop", "out_of_corpus"]


class GoldenQAPair(BaseModel):
    """One golden question, its expected answer, and the sources it should
    be traceable to (empty for out_of_corpus, where refusal is correct)."""

    model_config = ConfigDict(frozen=True)

    qa_id: str
    question: str
    question_language: Language
    expected_answer: str
    expected_source_doc_ids: list[str]
    expected_locators: list[str]
    category: GoldenCategory
    is_cross_lingual: bool
    notes: str | None = None

    @model_validator(mode="after")
    def _out_of_corpus_has_no_sources(self) -> GoldenQAPair:
        is_out_of_corpus = self.category == "out_of_corpus"
        has_sources = bool(self.expected_source_doc_ids) or bool(self.expected_locators)
        if is_out_of_corpus and has_sources:
            raise ValueError("out_of_corpus pairs must not carry expected sources")
        if not is_out_of_corpus and not has_sources:
            raise ValueError(f"{self.category} pairs must carry at least one expected source")
        return self
