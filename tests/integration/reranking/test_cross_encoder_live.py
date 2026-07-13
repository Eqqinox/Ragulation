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
