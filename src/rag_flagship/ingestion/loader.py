"""Loads already-parsed passages back from data/processed/*.jsonl.

Shared by scripts/build_index.py and the /ingest API route.
"""

from __future__ import annotations

from pathlib import Path

from rag_flagship.ingestion.models import CorpusPassage


def load_processed_passages(processed_dir: Path) -> list[CorpusPassage]:
    passages: list[CorpusPassage] = []
    for path in sorted(processed_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            passages.extend(CorpusPassage.model_validate_json(line) for line in handle)
    return passages
