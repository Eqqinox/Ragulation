"""Reorders retrieved candidates by cross-encoder relevance score.

The scorer is injected (a `Reranker`, structurally satisfied by a real
`sentence_transformers.CrossEncoder`) so this module never needs to load
the real ~500 MB model in a unit test; see tests/unit/reranking/.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from llama_index.core.schema import NodeWithScore

PairInput = tuple[str, str]


class Reranker(Protocol):
    def predict(self, inputs: list[PairInput]) -> Sequence[float]: ...


def rerank(
    query_text: str,
    candidates: Sequence[NodeWithScore],
    reranker: Reranker,
    top_k: int,
) -> list[NodeWithScore]:
    """Score every candidate against the query and return the top_k, highest
    cross-encoder score first. Each returned node's .score is overwritten
    with the cross-encoder score (the fusion score from retrieval is not
    comparable to it and is discarded)."""
    if not candidates:
        return []

    pairs: list[PairInput] = [(query_text, node.node.get_content()) for node in candidates]
    scores = reranker.predict(pairs)

    scored = [
        NodeWithScore(node=candidate.node, score=float(score))
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda node: node.score or float("-inf"), reverse=True)
    return scored[:top_k]
