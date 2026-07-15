# Ragulation

A retrieval-augmented generation assistant that answers questions over the
EU AI Act, GDPR, and related regulatory guidance, in English and French,
with a RAGAS-based evaluation pipeline that gates merges on measured
faithfulness and relevancy.

![License](https://img.shields.io/badge/license-Apache--2.0-blue)

## Status

Foundations, ingestion, retrieval tuning, generation, and evaluation
complete: corpus acquisition, Docling-based ingestion, three chunking
strategies (all indexed at full corpus scale), an Ollama `bge-m3`
embedding client, Qdrant hybrid indexing, a cross-encoder reranker,
Mistral generation with citations and refusal, a FastAPI service, a
62-pair golden QA dataset, and a RAGAS evaluation pipeline (faithfulness,
answer relevancy, context precision, context recall) with an automated
CI faithfulness gate. Verified end to end locally (fetch, ingest, chunk,
index, query, rerank, generate, serve, evaluate): 102 tests passing
(84 unit, 18 integration), 88% unit test coverage, clean `pip-audit` and
`gitleaks`. Lint/type/test CI is confirmed green on GitHub Actions
across the full Python 3.11/3.12/3.13 matrix; the CI faithfulness gate
is also confirmed green on real GitHub Actions runs.

## The problem it solves

Answering questions against EU AI Act and GDPR text, plus the guidance that
interprets it, by hand means cross-referencing dozens of articles, recitals,
and separately published guidelines. This project builds a Q&A assistant
that retrieves the relevant passages, cites them explicitly, and refuses to
answer when the retrieved context is insufficient, with its accuracy
measured rather than assumed.

## Quick start

```bash
git clone <repository-url>
cd ragulation
uv sync
docker compose up -d
ollama serve &
ollama pull bge-m3
ollama pull mistral-small3.2
uv run python scripts/build_index.py --strategy recursive
```

The corpus is already committed (`data/raw/`, `data/processed/`), so a
clean clone can index directly. To reproduce the fetch and parsing steps
from scratch instead:

```bash
uv run python scripts/fetch_corpus.py
uv run python scripts/ingest_corpus.py
```

Then run the API and ask a question:

```bash
uv run uvicorn rag_flagship.api.app:app --reload
curl -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the conditions for consent under GDPR?", "language": "en"}'
```

A response looks like:

```json
{
  "answer": "...",
  "sources": [{"doc_id": "gdpr_en", "locator": "Article 7 - Conditions for consent", "score": 0.98}],
  "refused": false
}
```

Questions with no basis in the corpus (for example, about traffic law)
return `"refused": true` instead of a fabricated answer.

## Key features

- Bilingual EU regulatory corpus (AI Act + GDPR official text in English
  and French, plus 12 curated EDPB/Commission/GPAI Code of Practice
  guidance documents, English) sourced directly from EUR-Lex and the
  Publications Office's Cellar repository.
- Docling-based ingestion recovering legal structure (articles, recitals,
  chapters, guidance sections) as individually citable passages.
- Three chunking strategies (recursive, semantic, parent-child)
  implemented behind one common interface, ready for a later comparative
  evaluation stage.
- Hybrid retrieval: dense `bge-m3` embeddings plus BM25, fused with
  Reciprocal Rank Fusion, served from a local Qdrant instance.
- Cross-encoder reranking (`BAAI/bge-reranker-v2-m3`) over the hybrid
  retrieval candidates before generation.
- Mistral (`mistral-small3.2` via Ollama) generation with a
  citation-and-refusal prompt: a two-layer refusal design (a fast,
  deterministic reranker-score threshold plus the model's own instructed
  fallback) and structural, independently-verifiable citations returned
  alongside every answer.
- A FastAPI service (`GET /health`, `POST /query`, `POST /ingest`) built
  on top of the same pipeline, with heavy dependencies (embedding model,
  reranker, LLM, vector store client) constructed once at startup.
- A 62-pair hand-curated golden question set (factual, multi-hop,
  out-of-corpus traps, and a cross-lingual subset), each fact-checked
  against the real corpus while authoring, and used to calibrate the
  refusal threshold against measured reranker scores.
- RAGAS evaluation (faithfulness, answer relevancy, context precision,
  context recall) across a 12-configuration comparison grid (3 chunking
  strategies x 2 retrieval modes x with/without reranker), plus an
  automated CI gate that blocks merges below a measured faithfulness
  floor, using a small judge model sized for GitHub-hosted runners.

## Architecture

Three sequential CLI scripts build the index, then a FastAPI service
answers questions against it, backed by a local Ollama server and a local
Qdrant Docker container:

```mermaid
flowchart LR
    A[fetch_corpus.py] --> B[data/raw/]
    B --> C[ingest_corpus.py]
    C --> D[data/processed/]
    D --> E[build_index.py]
    E --> F[(Qdrant)]
    F --> G[rag_flagship.api]
    G -->|"rerank"| H[bge-reranker-v2-m3]
    G -->|"generate"| I[Ollama mistral-small3.2]
    F --> J[run_eval_grid.py / run_ci_eval_gate.py]
    J -->|"score"| K[RAGAS: faithfulness, relevancy, precision, recall]
```

`src/rag_flagship/` is organized as one package per pipeline stage
(`corpus`, `ingestion`, `chunking`, `embeddings`, `indexing`, `reranking`,
`generation`, `api`, `evaluation`, `golden`), each with its own tests
under `tests/unit/` and `tests/integration/`.

## Usage

```bash
uv run python scripts/build_index.py --strategy {recursive,semantic,parent_child}
uv run python scripts/calibrate_refusal_threshold.py
uv run uvicorn rag_flagship.api.app:app --reload
uv run python scripts/run_eval_grid.py grid    # the 12-config RAGAS comparison
uv run pytest -q                    # unit tests, fast, no network or live models
uv run pytest -q -m integration     # integration tests, needs Ollama and Qdrant running
```

## Configuration

Every setting is typed and environment-driven (pydantic-settings). Copy
`.env.example` to `.env` and adjust:

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_DENSE_MODEL_NAME` | `bge-m3` | Dense embedding model |
| `OLLAMA_GENERATION_MODEL_NAME` | `mistral-small3.2` | Generation model |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant instance URL |
| `QDRANT_API_KEY` | empty | Qdrant API key, if required |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker model |
| `RERANKER_DEVICE` | `cpu` | Hardcoded, not auto-detected: GPU inference silently produced wrong scores under memory contention with Ollama on this machine |
| `OLLAMA_JUDGE_MODEL_NAME` | `mistral-small3.2` | RAGAS judge model; overridden to a small model in CI |

Ollama must additionally be started with `OLLAMA_CONTEXT_LENGTH=16384`
for evaluation to work correctly: its OpenAI-compatible endpoint
silently ignores smaller per-request context overrides, so the context
window has to be raised server-wide instead.

## Evaluation

RAGAS scores this pipeline (faithfulness, answer relevancy, context
precision, context recall) across a 12-configuration grid: 3 chunking
strategies (recursive, semantic, parent-child) x dense-only vs. hybrid
retrieval x with/without reranking, run against a fixed golden-dataset
subsample. The winning configuration -- semantic chunking with
dense-only retrieval -- was then re-evaluated against the full
56-question answerable set (`out_of_corpus` questions excluded, since
faithfulness-style metrics don't meaningfully apply to a question the
system should refuse) for a statistically sturdier headline number:

| Metric | Naive baseline (recursive, dense, no reranker) | Tuned winner, full 56-question set |
|---|---|---|
| Faithfulness | 0.900 | **0.915** |
| Answer relevancy | 0.763 | 0.782 |
| Context precision | 0.733 | 0.743 |
| Context recall | 0.944 | 0.943 |

The baseline column is measured on the 12-question subsample (only the
winning configuration was re-run at the full 56-question scale, since
re-running all 12 at full scale would take days); on that same
subsample the winner actually scored a perfect 1.000 faithfulness,
which dropped to 0.915 once measured against 4x more questions -- a
useful reminder that small-sample eval results can look better than
they are, and the fuller number is the one to trust.

The full 12-config grid is in `scripts/run_eval_grid.py`, runnable
locally end to end; results are also available as raw JSON for anyone
who wants to inspect every question's score rather than just the
means. A small, automated CI gate (`.github/workflows/eval.yml`) checks
faithfulness on every push and pull request using a compact judge model
sized for GitHub-hosted runners.

This CI gate takes roughly 30 minutes end to end, most of which is not
the RAGAS scoring itself: `BAAI/bge-reranker-v2-m3` (the cross-encoder
reranker, unrelated to the judge/generation models) takes a measured
~14 minutes to load on a GitHub-hosted runner's constrained CPU. This
was investigated directly, not assumed: caching its Hugging Face Hub
weights had no effect (confirmed via a cache hit that still showed the
same delay), and neither did `low_cpu_mem_usage=True` (a documented fix
for a different, longstanding `transformers` loading inefficiency).
Since this gate only checks correctness on every push, not the ability
to keep developing locally in the meantime, the wait is left as a known
characteristic rather than a bug to keep chasing.

## Key decisions

Fully open-source and local stack, chosen to run entirely on the
developer's own machine: Docling for parsing, LlamaIndex for chunking and
vector store orchestration, Ollama `bge-m3` for dense embeddings, BM25
(`fastembed`) for the sparse side of hybrid retrieval, Qdrant (self-hosted
via Docker) for storage and Reciprocal Rank Fusion, `sentence-transformers`
for cross-encoder reranking, and Ollama `mistral-small3.2` for generation.
Data is tracked directly in Git rather than with a data-versioning tool,
since the corpus is small enough not to need one. Citations are returned
as a structural, typed field on every response rather than left to the
model's own inline prose, since the latter is not reliably accurate.
Evaluation uses `ragas` with a judge and embeddings client built against
Ollama's OpenAI-compatible endpoint (`openai` as a thin HTTP client, not
the real OpenAI API), keeping the same 0 €, fully local stance end to
end; the CI gate uses a different, smaller judge model than local runs,
since the full-size generation model does not fit a GitHub-hosted
runner's disk budget.

## Security

See `SECURITY.md` for how to report a vulnerability. Retrieved and
generated content is treated as untrusted data throughout the pipeline.

## Contributing

See `CONTRIBUTING.md`.

## License

Apache License 2.0, see `LICENSE`.
