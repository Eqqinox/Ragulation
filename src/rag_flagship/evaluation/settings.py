"""Typed, environment-driven configuration for the RAGAS judge.

Kept separate from GenerationSettings even though both talk to the same
Ollama server: the judge model is a distinct concern from the
generation model. Locally both default to mistral-small3.2; the CI gate
overrides OLLAMA_JUDGE_MODEL_NAME to a small model sized for a
GitHub-hosted runner's disk budget. See ADR-0006.

Operational requirement, not a setting in this file: the Ollama server
itself must be started with OLLAMA_CONTEXT_LENGTH>=16384 (see
judge_max_tokens below for why). Confirmed directly: Ollama's OpenAI-
compatible endpoint (used here, see judge.py) silently ignores any
per-request context-window override sent by the client (top-level
num_ctx, and options.num_ctx via extra_body both tested, neither took
effect on Ollama 0.31.1 -- ollama ps kept reporting the server's
original default). Only Ollama's native /api/chat endpoint honors a
per-request override; since RAGAS's OpenAI-compatible integration is
the one non-deprecated path (see ADR-0006), the context window has to
be raised server-wide instead of per-request.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    base_url: str = "http://localhost:11434"
    judge_model_name: str = "mistral-small3.2"
    """See ADR-0006: mistral-small3.2 for local/headline runs. CI uses a
    different, smaller model, since GitHub-hosted runners cannot fit
    mistral-small3.2 (15 GB) within their ~14 GB disk budget."""
    judge_embedding_model_name: str = "bge-m3"
    """Used by RAGAS's AnswerRelevancy metric, which needs embeddings to
    compare the generated answer against synthetic questions derived
    from it. Reuses the same bge-m3 model already served by Ollama for
    retrieval (rag_flagship.embeddings), not a second embedding model."""
    judge_max_tokens: int = 4096
    """RAGAS's own InstructorModelArgs defaults max_tokens to 1024, which
    is too small for Faithfulness/ContextRecall's structured output
    against real retrieved passages: this measurably raised
    IncompleteOutputException (finish_reason="length") during a real
    grid run. Necessary but not sufficient on its own -- see the module
    docstring's OLLAMA_CONTEXT_LENGTH note: with Ollama's default 4096
    context window, ContextRecall's prompt (5 retrieved passages, the
    question, the reference answer, plus RAGAS's own structured-output
    schema instructions) fills the window before any max_tokens budget
    is even reached. Confirmed by testing every one of the four metrics
    individually against real retrieved passages, both before and after
    raising the server's context window."""
