"""Runs the semantic chunker against the real local Ollama bge-m3 model.

Run explicitly with: uv run pytest -m integration
Requires a local Ollama server with bge-m3 pulled (ollama pull bge-m3).
"""

import json
from pathlib import Path

import pytest
from llama_index.embeddings.ollama import OllamaEmbedding

from rag_flagship.chunking.semantic import SemanticChunker
from rag_flagship.ingestion.models import CorpusPassage

pytestmark = pytest.mark.integration

PROCESSED_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"


def _load_long_passage() -> CorpusPassage:
    with (PROCESSED_DIR / "gdpr_en.jsonl").open(encoding="utf-8") as handle:
        passages = [CorpusPassage.model_validate_json(line) for line in handle]
    return max(passages, key=lambda p: len(p.text))


def test_semantic_chunking_with_real_bge_m3() -> None:
    embed_model = OllamaEmbedding(model_name="bge-m3")
    chunker = SemanticChunker(embed_model=embed_model)
    passage = _load_long_passage()

    chunks = chunker.chunk([passage])

    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.text
        assert chunk.text in passage.text


def test_semantic_chunk_metadata_survives_json_roundtrip() -> None:
    embed_model = OllamaEmbedding(model_name="bge-m3")
    chunker = SemanticChunker(embed_model=embed_model)
    passage = _load_long_passage()

    chunks = chunker.chunk([passage])
    roundtripped = [json.loads(c.model_dump_json()) for c in chunks]

    assert all(c["strategy"] == "semantic" for c in roundtripped)
