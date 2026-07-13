import uuid

from rag_flagship.chunking.models import Chunk
from rag_flagship.indexing.pipeline import chunk_to_node

CHUNK = Chunk(
    chunk_id="gdpr_en:Article 5:recursive:0",
    doc_id="gdpr_en",
    language="en",
    category="regulation",
    locator="Article 5",
    strategy="recursive",
    text="Personal data shall be processed lawfully.",
)


def test_point_id_is_a_valid_uuid() -> None:
    node = chunk_to_node(CHUNK)

    parsed = uuid.UUID(node.node_id)
    assert str(parsed) == node.node_id


def test_point_id_is_deterministic_for_the_same_chunk_id() -> None:
    first = chunk_to_node(CHUNK)
    second = chunk_to_node(CHUNK.model_copy())

    assert first.node_id == second.node_id


def test_different_chunk_ids_map_to_different_point_ids() -> None:
    other = CHUNK.model_copy(update={"chunk_id": "gdpr_en:Article 6:recursive:0"})

    first = chunk_to_node(CHUNK)
    second = chunk_to_node(other)

    assert first.node_id != second.node_id


def test_node_metadata_preserves_chunk_fields() -> None:
    node = chunk_to_node(CHUNK)

    assert node.metadata["chunk_id"] == "gdpr_en:Article 5:recursive:0"
    assert node.metadata["doc_id"] == "gdpr_en"
    assert node.metadata["language"] == "en"
    assert node.metadata["category"] == "regulation"
    assert node.metadata["locator"] == "Article 5"
    assert node.metadata["strategy"] == "recursive"
    assert node.metadata["parent_chunk_id"] is None
    assert node.get_content() == CHUNK.text


def test_parent_chunk_id_is_preserved_when_set() -> None:
    child = CHUNK.model_copy(
        update={
            "chunk_id": "gdpr_en:Article 5:parent_child:1",
            "parent_chunk_id": "gdpr_en:Article 5:parent_child:0",
        }
    )

    node = chunk_to_node(child)

    assert node.metadata["parent_chunk_id"] == "gdpr_en:Article 5:parent_child:0"
