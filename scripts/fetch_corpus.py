"""CLI entry point to download the corpus manifest into data/raw/.

Usage:
    uv run python scripts/fetch_corpus.py
    uv run python scripts/fetch_corpus.py --force
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from rag_flagship.corpus.fetcher import fetch_all
from rag_flagship.corpus.manifest import CORPUS_MANIFEST

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "raw"


@app.command()
def fetch(
    dest: Path = DEFAULT_DEST,
    force: bool = typer.Option(False, help="Re-download files already present on disk."),
) -> None:
    """Fetch every document in the corpus manifest into DEST."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    results = fetch_all(CORPUS_MANIFEST, dest_dir=dest, force=force)
    downloaded = sum(1 for r in results if r.status == "downloaded")
    skipped = sum(1 for r in results if r.status == "skipped")
    typer.echo(f"Done: {downloaded} downloaded, {skipped} already present, {len(results)} total.")


if __name__ == "__main__":
    app()
