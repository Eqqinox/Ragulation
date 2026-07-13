import json
from dataclasses import dataclass, field

import pytest
import requests

from rag_flagship.corpus.fetcher import CorpusFetchError, fetch_all, fetch_document
from rag_flagship.corpus.manifest import CorpusDocument

SAMPLE_DOC = CorpusDocument(
    doc_id="sample_doc",
    title="Sample Document",
    url="https://example.invalid/sample",
    language="en",
    media_type="html",
    category="regulation",
    source_organization="Test",
    published="2026-01-01",
)


@dataclass
class FakeResponse:
    content: bytes
    status_code: int = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


@dataclass
class FakeSession:
    responses: list[object] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    received_headers: list[dict[str, str]] = field(default_factory=list)

    def get(self, url: str, timeout: float, headers: dict[str, str]) -> FakeResponse:
        self.calls.append(url)
        self.received_headers.append(headers)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, FakeResponse)
        return item


def test_fetch_document_downloads_and_writes_metadata(tmp_path) -> None:
    session = FakeSession(responses=[FakeResponse(content=b"<html>hello</html>")])

    result = fetch_document(SAMPLE_DOC, dest_dir=tmp_path, session=session)

    assert result.status == "downloaded"
    assert result.path.read_bytes() == b"<html>hello</html>"
    metadata = json.loads(result.path.with_name(result.path.name + ".meta.json").read_text())
    assert metadata["doc_id"] == "sample_doc"
    assert metadata["sha256"] == result.sha256


def test_fetch_document_forwards_document_request_headers(tmp_path) -> None:
    doc_with_headers = SAMPLE_DOC.model_copy(
        update={
            "doc_id": "sample_doc_with_headers",
            "request_headers": {"Accept-Language": "fra"},
        }
    )
    session = FakeSession(responses=[FakeResponse(content=b"bonjour")])

    fetch_document(doc_with_headers, dest_dir=tmp_path, session=session)

    assert session.received_headers == [{"Accept-Language": "fra"}]


def test_fetch_document_skips_existing_file_by_default(tmp_path) -> None:
    session = FakeSession(responses=[FakeResponse(content=b"first")])
    first = fetch_document(SAMPLE_DOC, dest_dir=tmp_path, session=session)
    assert first.status == "downloaded"

    second = fetch_document(SAMPLE_DOC, dest_dir=tmp_path, session=session)

    assert second.status == "skipped"
    assert second.sha256 == first.sha256
    assert session.calls == [str(SAMPLE_DOC.url)]


def test_fetch_document_force_redownloads(tmp_path) -> None:
    session = FakeSession(
        responses=[FakeResponse(content=b"first"), FakeResponse(content=b"second")]
    )
    fetch_document(SAMPLE_DOC, dest_dir=tmp_path, session=session)

    result = fetch_document(SAMPLE_DOC, dest_dir=tmp_path, session=session, force=True)

    assert result.status == "downloaded"
    assert result.path.read_bytes() == b"second"


def test_fetch_document_raises_after_exhausting_retries(tmp_path) -> None:
    session = FakeSession(
        responses=[
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
            requests.ConnectionError("boom"),
        ]
    )

    with pytest.raises(CorpusFetchError):
        fetch_document(SAMPLE_DOC, dest_dir=tmp_path, session=session)


def test_fetch_all_sleeps_between_downloads_but_not_after_the_last(tmp_path) -> None:
    other_doc = SAMPLE_DOC.model_copy(update={"doc_id": "sample_doc_2"})
    session = FakeSession(responses=[FakeResponse(content=b"a"), FakeResponse(content=b"b")])
    sleep_calls: list[float] = []

    fetch_all(
        [SAMPLE_DOC, other_doc],
        dest_dir=tmp_path,
        session=session,
        delay_seconds=0.01,
        sleep_fn=sleep_calls.append,
    )

    assert sleep_calls == [0.01]


def test_fetch_all_does_not_sleep_when_everything_is_skipped(tmp_path) -> None:
    session = FakeSession(responses=[FakeResponse(content=b"a")])
    fetch_document(SAMPLE_DOC, dest_dir=tmp_path, session=session)
    sleep_calls: list[float] = []

    fetch_all(
        [SAMPLE_DOC],
        dest_dir=tmp_path,
        session=session,
        delay_seconds=0.01,
        sleep_fn=sleep_calls.append,
    )

    assert sleep_calls == []
