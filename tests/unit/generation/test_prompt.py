from llama_index.core.schema import NodeWithScore, TextNode

from rag_flagship.generation.prompt import REFUSAL_PHRASE, build_messages


def _node(text: str, doc_id: str, locator: str) -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, metadata={"doc_id": doc_id, "locator": locator}),
        score=1.0,
    )


def test_returns_system_and_user_message() -> None:
    passages = [_node("Some text.", "gdpr_en", "Article 5")]

    messages = build_messages("What is X?", "en", passages)

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    assert messages[1].content == "What is X?"


def test_system_message_contains_the_refusal_phrase_for_the_question_language() -> None:
    passages = [_node("Some text.", "gdpr_en", "Article 5")]

    en_messages = build_messages("What is X?", "en", passages)
    fr_messages = build_messages("Qu'est-ce que X ?", "fr", passages)

    assert REFUSAL_PHRASE["en"] in en_messages[0].content
    assert REFUSAL_PHRASE["fr"] in fr_messages[0].content


def test_system_message_instructs_citation_format() -> None:
    messages = build_messages("What is X?", "en", [_node("text", "doc", "loc")])

    assert "[doc_id, locator]" in messages[0].content


def test_context_block_includes_doc_id_and_locator_for_each_passage() -> None:
    passages = [
        _node("First passage.", "gdpr_en", "Article 15"),
        _node("Second passage.", "ai_act_en", "Article 5"),
    ]

    messages = build_messages("Question?", "en", passages)

    assert "gdpr_en" in messages[0].content
    assert "Article 15" in messages[0].content
    assert "ai_act_en" in messages[0].content
    assert "Article 5" in messages[0].content


def test_empty_passages_still_produces_a_valid_prompt() -> None:
    messages = build_messages("Question?", "en", [])

    assert len(messages) == 2
    assert REFUSAL_PHRASE["en"] in messages[0].content
