"""Builds RAGAS-ready evaluation samples from golden pairs and pipeline
output.

Kept pure (no model or network calls) so it is directly unit-testable:
this module only reshapes data already produced elsewhere
(rag_flagship.golden, rag_flagship.generation, rag_flagship.indexing)
into the fields RAGAS's metrics expect.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rag_flagship.golden.models import GoldenQAPair


class EvalSample(BaseModel):
    """One golden question paired with this pipeline's actual output for
    a given configuration, ready to score against the four RAGAS metrics
    named in projet_1_rag_flagship.md's Semaine 3 scope."""

    model_config = ConfigDict(frozen=True)

    qa_id: str
    question: str
    answer: str
    retrieved_contexts: list[str]
    reference: str


def build_eval_sample(
    golden: GoldenQAPair, answer: str, retrieved_contexts: list[str]
) -> EvalSample:
    return EvalSample(
        qa_id=golden.qa_id,
        question=golden.question,
        answer=answer,
        retrieved_contexts=retrieved_contexts,
        reference=golden.expected_answer,
    )
