"""Runs the real BAAI/bge-reranker-v2-m3 cross-encoder.

Run explicitly with: uv run pytest -m integration
First run downloads the model (~500 MB) from Hugging Face.
"""

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from rag_flagship.reranking.cross_encoder import build_reranker
from rag_flagship.reranking.pipeline import rerank

pytestmark = pytest.mark.integration


def test_real_reranker_ranks_relevant_passage_above_unrelated_one() -> None:
    reranker = build_reranker()
    candidates = [
        NodeWithScore(
            node=TextNode(text="The weather today is sunny and warm."),
            score=0.5,
        ),
        NodeWithScore(
            node=TextNode(
                text="The data subject has the right to obtain confirmation "
                "as to whether personal data concerning them are processed."
            ),
            score=0.5,
        ),
    ]

    results = rerank("What is the right of access under the GDPR?", candidates, reranker, top_k=2)

    assert "data subject" in results[0].node.get_content()
    assert results[0].score is not None
    assert results[1].score is not None
    assert results[0].score > results[1].score


def test_scores_are_distinct_across_repeated_calls_on_the_same_instance() -> None:
    """Regression test for a real, silent failure found on this machine:
    letting sentence-transformers auto-select MPS (Apple's GPU backend)
    for this reranker produced identical scores for genuinely different
    passages (an Insufficient Memory Metal error in stderr, swallowed
    rather than raised) whenever Ollama's models were also resident on
    the GPU. Forcing device="cpu" (RerankerSettings default, see
    ADR-0004) fixed it. This test calls predict three times on one
    reranker instance and checks the scores are never all identical."""
    reranker = build_reranker()
    candidates = [
        NodeWithScore(node=TextNode(text="Cats sleep most of the day."), score=0.5),
        NodeWithScore(
            node=TextNode(text="Controllers must notify a breach within 72 hours."),
            score=0.5,
        ),
        NodeWithScore(node=TextNode(text="The weather in Paris is mild in spring."), score=0.5),
    ]

    for _ in range(3):
        results = rerank("data breach notification deadline", candidates, reranker, top_k=3)
        scores = {round(r.score, 6) for r in results if r.score is not None}
        assert len(scores) > 1, "all scores identical: the MPS silent-failure bug may have recurred"
