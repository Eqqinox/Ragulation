"""Builds the citation-and-refusal prompt. Pure: no model or network calls.

Follows the four-part 2026 RAG prompting pattern (ADR-0005): authority
(context overrides the model's own knowledge), scope (answer only from
the passages), constraints (an exact citation format), fallback (an
exact refusal phrase, in the question's own language).
"""

from __future__ import annotations

from collections.abc import Sequence

from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.schema import NodeWithScore

from rag_flagship.corpus.manifest import Language

REFUSAL_PHRASE: dict[Language, str] = {
    "en": "I cannot answer this from the provided context.",
    "fr": "Je ne peux pas repondre a partir du contexte fourni.",
}

_LANGUAGE_NAME: dict[Language, str] = {"en": "English", "fr": "French"}

_SYSTEM_TEMPLATE = """You are a compliance assistant answering questions about \
the EU AI Act, GDPR, and their official interpretive guidance.

Rules:
1. Answer only using the context passages below. Do not use outside \
knowledge, even if you know more about the topic.
2. Cite every factual claim inline in the format [doc_id, locator], \
matching one of the passages below exactly.
3. Answer in {language_name}, the same language as the question.
4. If the passages do not contain enough information to answer the \
question, respond with exactly this sentence and nothing else: \
"{refusal_phrase}"

Context passages:
{context_block}"""


def _format_passage(index: int, node: NodeWithScore) -> str:
    doc_id = node.node.metadata.get("doc_id", "unknown")
    locator = node.node.metadata.get("locator", "unknown")
    return f"[{index}] ({doc_id}, {locator}): {node.node.get_content()}"


def build_messages(
    question: str,
    question_language: Language,
    passages: Sequence[NodeWithScore],
) -> list[ChatMessage]:
    context_block = "\n\n".join(
        _format_passage(index, node) for index, node in enumerate(passages, start=1)
    )
    system_content = _SYSTEM_TEMPLATE.format(
        language_name=_LANGUAGE_NAME[question_language],
        refusal_phrase=REFUSAL_PHRASE[question_language],
        context_block=context_block,
    )
    return [
        ChatMessage(role=MessageRole.SYSTEM, content=system_content),
        ChatMessage(role=MessageRole.USER, content=question),
    ]
