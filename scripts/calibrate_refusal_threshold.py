"""Measures the cross-encoder score gap between out-of-corpus and other
golden questions, to pick the refusal threshold in
rag_flagship.generation.pipeline.REFUSAL_SCORE_THRESHOLD (see ADR-0005).

Usage:
    uv run python scripts/calibrate_refusal_threshold.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import typer

from rag_flagship.embeddings.dense import build_dense_embedding_model
from rag_flagship.golden.models import GoldenQAPair
from rag_flagship.indexing.pipeline import hybrid_query
from rag_flagship.indexing.store import build_vector_store
from rag_flagship.reranking.cross_encoder import build_reranker
from rag_flagship.reranking.pipeline import rerank

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "qa_v1.jsonl"


def _load_golden() -> list[GoldenQAPair]:
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        return [GoldenQAPair.model_validate_json(line) for line in handle]


@app.command()
def calibrate(collection: str = "rag_flagship_recursive", retrieval_top_k: int = 20) -> None:
    embed_model = build_dense_embedding_model()
    vector_store = build_vector_store(collection)
    reranker = build_reranker()

    out_of_corpus_scores: list[float] = []
    other_scores: list[float] = []

    for pair in _load_golden():
        candidates = hybrid_query(vector_store, embed_model, pair.question, top_k=retrieval_top_k)
        if not candidates:
            top_score = float("-inf")
        else:
            reranked = rerank(pair.question, candidates, reranker, top_k=1)
            top_score = reranked[0].score if reranked[0].score is not None else float("-inf")

        if pair.category == "out_of_corpus":
            out_of_corpus_scores.append(top_score)
        else:
            other_scores.append(top_score)

        typer.echo(f"{pair.qa_id} [{pair.category}]: top_score={top_score:.3f}")

    typer.echo("")
    typer.echo(
        f"out_of_corpus: n={len(out_of_corpus_scores)}, "
        f"min={min(out_of_corpus_scores):.3f}, max={max(out_of_corpus_scores):.3f}, "
        f"mean={statistics.mean(out_of_corpus_scores):.3f}"
    )
    typer.echo(
        f"other: n={len(other_scores)}, "
        f"min={min(other_scores):.3f}, max={max(other_scores):.3f}, "
        f"mean={statistics.mean(other_scores):.3f}"
    )
    typer.echo(
        json.dumps(
            {
                "out_of_corpus_max": max(out_of_corpus_scores),
                "other_min": min(other_scores),
                "suggested_threshold": (max(out_of_corpus_scores) + min(other_scores)) / 2,
            }
        )
    )


if __name__ == "__main__":
    app()
