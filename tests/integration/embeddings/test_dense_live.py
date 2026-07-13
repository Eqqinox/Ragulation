"""Runs the real Ollama bge-m3 model.

Run explicitly with: uv run pytest -m integration
Requires a local Ollama server with bge-m3 pulled (ollama pull bge-m3).
"""

import numpy as np
import pytest

from rag_flagship.embeddings.dense import build_dense_embedding_model

pytestmark = pytest.mark.integration


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def test_bge_m3_returns_a_1024_dim_vector() -> None:
    model = build_dense_embedding_model()

    vector = model.get_text_embedding("The data subject has the right of access.")

    assert len(vector) == 1024


def test_bge_m3_is_multilingual_en_fr_similarity() -> None:
    model = build_dense_embedding_model()
    en = np.array(model.get_text_embedding("the right to erasure"))
    fr = np.array(model.get_text_embedding("le droit a l'effacement"))
    unrelated = np.array(model.get_text_embedding("the weather is sunny today"))

    assert _cosine(en, fr) > _cosine(en, unrelated)
