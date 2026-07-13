"""Live network checks against the real source servers.

Run explicitly with: uv run pytest -m integration
Not run by default; these depend on external services being reachable.
"""

import pytest

from rag_flagship.corpus.fetcher import fetch_document
from rag_flagship.corpus.manifest import CORPUS_MANIFEST

pytestmark = pytest.mark.integration


def test_gdpr_english_text_is_reachable(tmp_path) -> None:
    document = next(doc for doc in CORPUS_MANIFEST if doc.doc_id == "gdpr_en")

    result = fetch_document(document, dest_dir=tmp_path, force=True)

    assert result.status == "downloaded"
    assert result.path.stat().st_size > 10_000
