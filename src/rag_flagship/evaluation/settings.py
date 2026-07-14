"""Typed, environment-driven configuration for the RAGAS judge.

Kept separate from GenerationSettings even though both may talk to the
same Ollama server (or Ollama Cloud): the judge model is a distinct
concern from the generation model. Locally both default to
mistral-small3.2 against a local Ollama server. The CI gate points
judge_base_url/judge_api_key (and GenerationSettings'
generation_base_url/generation_api_key) at Ollama Cloud instead, using
gemma4 -- a small local judge (first mistral-small3.2, then
qwen2.5:1.5b, qwen2.5:3b, and ministral-3:3b) proved unreliable on the
GitHub-hosted runner's CPU-only Ollama build specifically for RAGAS's
ContextRecall metric, every time, even after separately verifying each
one reliable on this project's local GPU-accelerated Ollama first; see
ADR-0006's addenda for the full history. Ollama Cloud runs on real
inference hardware, removing that backend-dependent gap rather than
continuing to search for a small model immune to it. Its catalog has no
embedding models, though, so bge-m3 embeddings still run on a locally
installed Ollama in CI (see EmbeddingSettings) -- only generation and
the judge move to the cloud.

Operational requirement for a *local* Ollama server specifically (not
Ollama Cloud, which manages its own context window): it must be started
with OLLAMA_CONTEXT_LENGTH>=16384 (see judge_max_tokens below for why).
Confirmed directly: Ollama's OpenAI-compatible endpoint (used here, see
judge.py) silently ignores any per-request context-window override sent
by the client (top-level num_ctx, and options.num_ctx via extra_body
both tested, neither took effect on Ollama 0.31.1 -- ollama ps kept
reporting the server's original default). Only Ollama's native
/api/chat endpoint honors a per-request override; since RAGAS's
OpenAI-compatible integration is the one non-deprecated path (see
ADR-0006), a local server's context window has to be raised server-wide
instead of per-request.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class EvaluationSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    judge_base_url: str = "http://localhost:11434"
    """Deliberately its own field, not shared with EmbeddingSettings'
    base_url: the CI faithfulness gate points the judge at Ollama Cloud
    while keeping embeddings local (Ollama Cloud's catalog has no
    embedding models -- see the addendum below), so the two must be
    independently overridable via distinct env vars in the same job."""
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
    judge_api_key: str = ""
    """Empty for a local Ollama server (no auth). Set alongside
    judge_base_url=https://ollama.com to judge via Ollama Cloud instead
    -- used by the CI faithfulness gate after ministral-3:3b, verified
    reliable on this project's local GPU-accelerated Ollama, still
    failed RAGAS's ContextRecall on the GitHub-hosted runner's CPU-only
    Ollama build. Ollama Cloud runs on real inference hardware, not the
    runner's own CPU, eliminating that backend-dependent gap entirely
    rather than continuing to hunt for a small model immune to it. Its
    model catalog is large frontier models only (no embedding models),
    so the CI gate keeps bge-m3 embeddings on a locally installed Ollama
    while only generation and the judge move to the cloud. See
    ADR-0006's addendum."""
