"""Download the corpus manifest to disk.

Fetches are one request per document (see ``manifest.py``), not bulk
scraping. A descriptive User-Agent and a delay between requests keep this
polite to the source servers. Reuse under Commission Decision 2011/833/EU
is recorded in ADR-0001.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from rag_flagship.corpus.manifest import CorpusDocument

logger = logging.getLogger(__name__)

USER_AGENT = "p1-rag-flagship-corpus-fetcher/0.1 (contact: meknaci81@gmail.com)"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_DELAY_SECONDS = 1.5


class CorpusFetchError(Exception):
    """Raised when a corpus document cannot be fetched or written to disk."""


class HttpGetter(Protocol):
    def get(self, url: str, timeout: float, headers: dict[str, str]) -> requests.Response: ...


@dataclass(frozen=True)
class FetchResult:
    document: CorpusDocument
    path: Path
    sha256: str
    status: Literal["downloaded", "skipped"]


def default_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _extension(document: CorpusDocument) -> str:
    return "html" if document.media_type == "html" else "pdf"


def raw_file_path(document: CorpusDocument, dest_dir: Path) -> Path:
    """Where a document's raw bytes live (or will live) under dest_dir."""
    return dest_dir / document.language / f"{document.doc_id}.{_extension(document)}"


@retry(
    retry=retry_if_exception_type(requests.RequestException),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _get_with_retry(
    session: requests.Session | HttpGetter,
    url: str,
    timeout: float,
    headers: dict[str, str],
) -> requests.Response:
    response = session.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return response


def fetch_document(
    document: CorpusDocument,
    dest_dir: Path,
    session: requests.Session | HttpGetter | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    force: bool = False,
) -> FetchResult:
    """Fetch a single document, or reuse the file already on disk."""
    destination = raw_file_path(document, dest_dir)

    if destination.exists() and not force:
        existing = destination.read_bytes()
        return FetchResult(
            document=document,
            path=destination,
            sha256=hashlib.sha256(existing).hexdigest(),
            status="skipped",
        )

    active_session = session if session is not None else default_session()
    try:
        response = _get_with_retry(
            active_session, str(document.url), timeout, document.request_headers
        )
    except requests.RequestException as exc:
        raise CorpusFetchError(
            f"failed to fetch {document.doc_id} from {document.url}: {exc}"
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    checksum = hashlib.sha256(response.content).hexdigest()

    metadata_path = destination.with_name(destination.name + ".meta.json")
    metadata_path.write_text(
        json.dumps(
            {
                "doc_id": document.doc_id,
                "url": str(document.url),
                "sha256": checksum,
                "fetched_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )

    return FetchResult(document=document, path=destination, sha256=checksum, status="downloaded")


def fetch_all(
    manifest: Iterable[CorpusDocument],
    dest_dir: Path,
    session: requests.Session | HttpGetter | None = None,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    force: bool = False,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[FetchResult]:
    """Fetch every document in the manifest, with a delay between live requests."""
    active_session = session if session is not None else default_session()
    documents = list(manifest)
    results: list[FetchResult] = []
    for index, document in enumerate(documents):
        result = fetch_document(document, dest_dir, session=active_session, force=force)
        results.append(result)
        logger.info("%s: %s (%s)", document.doc_id, result.status, result.path)
        if result.status == "downloaded" and index < len(documents) - 1:
            sleep_fn(delay_seconds)
    return results
