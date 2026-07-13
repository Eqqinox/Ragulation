"""CLI entry point to parse every fetched raw document into data/processed/.

Usage:
    uv run python scripts/ingest_corpus.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from docling.document_converter import DocumentConverter

from rag_flagship.corpus.manifest import CORPUS_MANIFEST
from rag_flagship.ingestion.pipeline import ingest_document

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"


@app.command()
def ingest(
    raw_dir: Path = DEFAULT_RAW_DIR,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
) -> None:
    """Parse every document in the corpus manifest into cited passages."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    processed_dir.mkdir(parents=True, exist_ok=True)
    converter = DocumentConverter()
    total_passages = 0

    for document in CORPUS_MANIFEST:
        passages = ingest_document(document, raw_dir=raw_dir, converter=converter)
        out_path = processed_dir / f"{document.doc_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as handle:
            for passage in passages:
                handle.write(passage.model_dump_json() + "\n")
        typer.echo(f"{document.doc_id}: {len(passages)} passages -> {out_path}")
        total_passages += len(passages)

    typer.echo(f"Done: {total_passages} passages across {len(CORPUS_MANIFEST)} documents.")


if __name__ == "__main__":
    app()
