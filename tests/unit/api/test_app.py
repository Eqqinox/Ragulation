from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient
from llama_index.core.base.llms.types import ChatMessage, ChatResponse, MessageRole
from llama_index.core.schema import NodeWithScore, TextNode

import rag_flagship.api.app as app_module
from rag_flagship.api.app import AppState, app, get_state
from rag_flagship.chunking.models import Chunk
from rag_flagship.ingestion.models import CorpusPassage


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


@dataclass
class FakeQdrantClient:
    healthy: bool = True

    def get_collections(self) -> object:
        if not self.healthy:
            raise RuntimeError("qdrant is down")
        return object()

    def close(self) -> None:
        pass


@dataclass
class FakeReranker:
    def predict(self, inputs: list[tuple[str, str]]) -> list[float]:
        return [1.0] * len(inputs)


@dataclass
class FakeLLM:
    content: str = "Cited answer [gdpr_en, Article 5]."

    def chat(self, messages: list[ChatMessage]) -> ChatResponse:
        return ChatResponse(message=ChatMessage(role=MessageRole.ASSISTANT, content=self.content))


def _state(qdrant_healthy: bool = True, ollama_base_url: str = "http://ollama.invalid") -> AppState:
    return AppState(
        embed_model=object(),  # type: ignore[arg-type]
        reranker=FakeReranker(),  # type: ignore[arg-type]
        llm=FakeLLM(),  # type: ignore[arg-type]
        qdrant_client=FakeQdrantClient(healthy=qdrant_healthy),  # type: ignore[arg-type]
        ollama_base_url=ollama_base_url,
    )


client = TestClient(app)


def test_health_reports_ok_when_both_services_reachable(monkeypatch: pytest.MonkeyPatch) -> None:
    app.dependency_overrides[get_state] = lambda: _state(qdrant_healthy=True)
    monkeypatch.setattr(
        app_module.httpx2, "get", lambda *a, **k: type("R", (), {"status_code": 200})()
    )

    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["qdrant_reachable"] is True
    assert body["ollama_reachable"] is True


def test_health_reports_degraded_when_qdrant_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    app.dependency_overrides[get_state] = lambda: _state(qdrant_healthy=False)
    monkeypatch.setattr(
        app_module.httpx2, "get", lambda *a, **k: type("R", (), {"status_code": 200})()
    )

    response = client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert body["qdrant_reachable"] is False


def test_health_reports_degraded_when_ollama_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx2

    app.dependency_overrides[get_state] = lambda: _state(qdrant_healthy=True)

    def raise_connect_error(*args: object, **kwargs: object) -> object:
        raise httpx2.ConnectError("refused")

    monkeypatch.setattr(app_module.httpx2, "get", raise_connect_error)

    response = client.get("/health")

    body = response.json()
    assert body["status"] == "degraded"
    assert body["ollama_reachable"] is False


def test_query_rejects_unknown_strategy() -> None:
    app.dependency_overrides[get_state] = lambda: _state()

    response = client.post("/query", json={"question": "What is X?", "strategy": "bogus"})

    assert response.status_code == 400


def test_query_returns_answer_with_structured_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    app.dependency_overrides[get_state] = lambda: _state()
    candidate = NodeWithScore(
        node=TextNode(text="relevant text", metadata={"doc_id": "gdpr_en", "locator": "Article 5"}),
        score=0.9,
    )
    monkeypatch.setattr(app_module, "hybrid_query", lambda *a, **k: [candidate])
    monkeypatch.setattr(app_module, "build_vector_store", lambda *a, **k: object())

    response = client.post("/query", json={"question": "What does Article 5 say?"})

    assert response.status_code == 200
    body = response.json()
    assert body["refused"] is False
    assert body["sources"][0]["doc_id"] == "gdpr_en"
    assert body["sources"][0]["locator"] == "Article 5"


def test_query_rejects_question_that_is_too_long() -> None:
    app.dependency_overrides[get_state] = lambda: _state()

    response = client.post("/query", json={"question": "x" * 3000})

    assert response.status_code == 422


def test_ingest_rejects_unknown_strategy() -> None:
    app.dependency_overrides[get_state] = lambda: _state()

    response = client.post("/ingest", json={"strategy": "bogus"})

    assert response.status_code == 400


@dataclass
class FakeChunker:
    chunks: list[Chunk] = field(default_factory=list)

    def chunk(self, passages: list[CorpusPassage]) -> list[Chunk]:
        return self.chunks


def test_ingest_returns_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    app.dependency_overrides[get_state] = lambda: _state()
    passage = CorpusPassage(
        doc_id="gdpr_en", language="en", category="regulation", locator="Article 5", text="text"
    )
    chunk = Chunk(
        chunk_id="gdpr_en:Article 5:recursive:0",
        doc_id="gdpr_en",
        language="en",
        category="regulation",
        locator="Article 5",
        strategy="recursive",
        text="text",
    )
    monkeypatch.setattr(app_module, "load_processed_passages", lambda *a, **k: [passage])
    monkeypatch.setattr(app_module, "build_chunker", lambda *a, **k: FakeChunker(chunks=[chunk]))
    monkeypatch.setattr(app_module, "build_vector_store", lambda *a, **k: object())
    monkeypatch.setattr(app_module, "index_chunks", lambda *a, **k: 1)

    response = client.post("/ingest", json={"strategy": "recursive"})

    assert response.status_code == 200
    body = response.json()
    assert body["passages"] == 1
    assert body["chunks_indexed"] == 1
    assert body["collection"] == "rag_flagship_recursive"
