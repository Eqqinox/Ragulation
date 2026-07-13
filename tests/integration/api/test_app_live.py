"""Runs the real FastAPI app end to end, through actual HTTP handling.

Run explicitly with: uv run pytest -m integration
Requires a local Qdrant instance (docker compose up -d, with
rag_flagship_recursive already indexed) and a local Ollama server with
bge-m3 and mistral-small3.2 pulled. The real lifespan runs here (loads
the reranker model and connects to Qdrant/Ollama), unlike the unit tests.
"""

import pytest
from fastapi.testclient import TestClient

from rag_flagship.api.app import app

pytestmark = pytest.mark.integration


def test_health_is_ok_against_the_real_local_stack() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant_reachable"] is True
    assert body["ollama_reachable"] is True


def test_query_endpoint_answers_a_known_good_question() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": (
                    "Within how many hours must a controller notify a personal "
                    "data breach to the supervisory authority, where feasible?"
                ),
                "language": "en",
                "strategy": "recursive",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert len(body["sources"]) > 0


def test_query_endpoint_refuses_out_of_corpus_question() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={
                "question": (
                    "What are the specific data breach notification deadlines "
                    "under the California Consumer Privacy Act (CCPA)?"
                ),
                "language": "en",
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is True
    assert body["sources"] == []
