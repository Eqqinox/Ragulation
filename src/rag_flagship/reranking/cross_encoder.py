"""Factory for the production cross-encoder reranker."""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from rag_flagship.reranking.settings import RerankerSettings


def build_reranker(settings: RerankerSettings | None = None) -> CrossEncoder:
    active_settings = settings if settings is not None else RerankerSettings()
    model: CrossEncoder = CrossEncoder(
        active_settings.model_name,
        device=active_settings.device,
        # transformers' from_pretrained() otherwise instantiates the full
        # model with random weights before overwriting them with the real
        # checkpoint -- a well-documented inefficiency (huggingface/
        # transformers#21913, #9205) that a real CI run showed costing
        # ~14 of this reranker's ~26-minute load-and-score time on a
        # GitHub-hosted runner's constrained CPU, unaffected by whether
        # the weights were freshly downloaded or already cache-hit. See
        # ADR-0006's addendum for the measured evidence.
        model_kwargs={"low_cpu_mem_usage": True},
    )
    return model
