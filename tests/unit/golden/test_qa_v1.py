import json
from pathlib import Path

from rag_flagship.corpus.manifest import CORPUS_MANIFEST
from rag_flagship.golden.models import GoldenQAPair

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "qa_v1.jsonl"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
KNOWN_DOC_IDS = {doc.doc_id for doc in CORPUS_MANIFEST}


def _load_pairs() -> list[GoldenQAPair]:
    with GOLDEN_PATH.open(encoding="utf-8") as handle:
        return [GoldenQAPair.model_validate_json(line) for line in handle]


def _load_known_locators() -> set[tuple[str, str]]:
    known: set[tuple[str, str]] = set()
    for path in PROCESSED_DIR.glob("*.jsonl"):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                known.add((record["doc_id"], record["locator"]))
    return known


def test_every_line_parses_as_a_valid_golden_qa_pair() -> None:
    pairs = _load_pairs()
    assert len(pairs) > 0


def test_pair_count_is_within_the_50_to_80_target() -> None:
    pairs = _load_pairs()
    assert 50 <= len(pairs) <= 80


def test_qa_ids_are_unique() -> None:
    pairs = _load_pairs()
    ids = [p.qa_id for p in pairs]
    assert len(ids) == len(set(ids))


def test_all_three_categories_are_represented() -> None:
    pairs = _load_pairs()
    categories = {p.category for p in pairs}
    assert categories == {"factual", "multi_hop", "out_of_corpus"}


def test_both_languages_are_represented() -> None:
    pairs = _load_pairs()
    languages = {p.question_language for p in pairs}
    assert languages == {"en", "fr"}


def test_cross_lingual_subset_is_non_empty_and_all_flagged_correctly() -> None:
    pairs = _load_pairs()
    cross_lingual = [p for p in pairs if p.is_cross_lingual]
    assert len(cross_lingual) >= 5
    for pair in cross_lingual:
        assert pair.expected_source_doc_ids


def test_out_of_corpus_pairs_have_no_expected_sources() -> None:
    pairs = _load_pairs()
    for pair in pairs:
        if pair.category == "out_of_corpus":
            assert pair.expected_source_doc_ids == []
            assert pair.expected_locators == []


def test_non_out_of_corpus_pairs_reference_known_doc_ids() -> None:
    pairs = _load_pairs()
    for pair in pairs:
        if pair.category != "out_of_corpus":
            for doc_id in pair.expected_source_doc_ids:
                assert doc_id in KNOWN_DOC_IDS, f"{pair.qa_id} references unknown doc_id {doc_id!r}"


def test_expected_locators_exist_in_the_processed_corpus() -> None:
    """Catches a typo'd locator or a locator that no longer exists after a
    re-run of the ingestion pipeline (see documentations/mapping/modules/golden.md)."""
    pairs = _load_pairs()
    known_locators = _load_known_locators()
    for pair in pairs:
        for doc_id, locator in zip(
            pair.expected_source_doc_ids, pair.expected_locators, strict=True
        ):
            assert (doc_id, locator) in known_locators, (
                f"{pair.qa_id}: ({doc_id!r}, {locator!r}) not found in data/processed/"
            )


def test_multi_hop_pairs_reference_at_least_two_locators() -> None:
    pairs = _load_pairs()
    for pair in pairs:
        if pair.category == "multi_hop":
            assert len(pair.expected_locators) >= 2, pair.qa_id
