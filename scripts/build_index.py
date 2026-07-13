"""CLI entry point to chunk and index the processed corpus into Qdrant.

Usage:
    uv run python scripts/build_index.py --strategy recursive
    uv run python scripts/build_index.py --strategy semantic
    uv run python scripts/build_index.py --strategy parent_child

Requires a local Qdrant instance (docker compose up -d) and a local
Ollama server with bge-m3 pulled (ollama pull bge-m3).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from rag_flagship.chunking.factory import build_chunker
from rag_flagship.embeddings.dense import build_dense_embedding_model
from rag_flagship.indexing.pipeline import index_chunks
from rag_flagship.indexing.store import build_vector_store, collection_name_for_strategy
from rag_flagship.ingestion.loader import load_processed_passages

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@app.command()
def build(
    strategy: str = typer.Option("recursive", help="recursive, semantic, or parent_child"),
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    leaf_only: bool = typer.Option(
        True, help="For parent_child, index only leaf chunks (skip parent blocks)."
    ),
) -> None:
    """Chunk every processed passage and index it into Qdrant."""
    embed_model = build_dense_embedding_model()
    chunker = build_chunker(strategy, embed_model)

    passages = load_processed_passages(processed_dir)
    chunks = chunker.chunk(passages)
    if strategy == "parent_child" and leaf_only:
        chunks = [c for c in chunks if c.parent_chunk_id is not None]

    collection_name = collection_name_for_strategy(strategy)
    vector_store = build_vector_store(collection_name)
    indexed_count = index_chunks(chunks, vector_store=vector_store, embed_model=embed_model)

    typer.echo(
        json.dumps(
            {
                "strategy": strategy,
                "collection": collection_name,
                "passages": len(passages),
                "chunks_indexed": indexed_count,
            }
        )
    )


if __name__ == "__main__":
    app()
