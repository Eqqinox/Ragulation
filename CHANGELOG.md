# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `rag_flagship.evaluation` module: RAGAS's four core metrics
  (faithfulness, answer relevancy, context precision, context recall),
  built on a judge/embeddings client pointed at Ollama's OpenAI-
  compatible endpoint.
- `rag_flagship.indexing.dense_query`: dense-only retrieval alongside
  the existing hybrid `hybrid_query`, for comparing dense-only versus
  hybrid retrieval.
- `scripts/run_eval_grid.py`: a 12-config comparison grid (3 chunking
  strategies x 2 retrieval modes x with/without reranker) against a
  fixed golden-dataset subsample, plus a `full` command producing the
  headline numbers for the winning configuration.
- `scripts/run_ci_eval_gate.py` and `.github/workflows/eval.yml`: an
  automated CI faithfulness gate running on every push and PR, using a
  small, GitHub-hosted-runner-sized judge model against a tiny fixed
  corpus subset.
- `semantic` and `parent_child` chunking strategies now indexed at full
  corpus scale (1597 passages -> 2968 and 3917 chunks respectively),
  alongside `recursive`.
- `ragas`, `openai` dependencies (`langchain`/`langchain-community` are
  also now direct dependencies, required transitively by `ragas`).
- `rag_flagship.reranking` module: cross-encoder reranking with
  `BAAI/bge-reranker-v2-m3` via `sentence-transformers`, CPU-only by
  design (see Fixed, below).
- `rag_flagship.generation` module: Mistral (`mistral-small3.2` via
  Ollama) answer generation with a citation-and-refusal prompt and a
  two-layer refusal design (a fast reranker-score threshold, calibrated
  at 0.27 against the full golden dataset, plus the model's own
  instructed fallback); structural, independently-verifiable citations
  (`answer`, `sources`, `refused` on every response).
- `rag_flagship.api` module: a FastAPI service (`GET /health`,
  `POST /query`, `POST /ingest`), heavy dependencies constructed once at
  startup; manually verified end to end against the real local stack,
  including a correct in-corpus answer with citations and a correct
  refusal on an out-of-corpus question.
- `scripts/calibrate_refusal_threshold.py`: measures the reranker score
  gap between out-of-corpus and in-corpus golden questions.
- Reranking, generation, and API dependencies (`sentence-transformers`,
  `llama-index-llms-ollama`, `fastapi`, `uvicorn[standard]`, `httpx2`).
- Repository scaffold: `src` layout, internal engineering documentation
  and audit trail, CI workflow placeholders.
- Decision to source the corpus from EU AI Act + GDPR (EN/FR, official
  EUR-Lex texts), plus curated EDPB and AI Act guidance documents.
- Decision on the core stack: Docling, LlamaIndex, Ollama `bge-m3`,
  self-hosted Qdrant, DVC.
- Core and dev dependencies (Docling, LlamaIndex, Qdrant client and vector
  store integration, fastembed, pydantic-settings, ruff, mypy, pytest,
  hypothesis, pre-commit, pip-audit).
- ADR-0003: drop DVC after `pip-audit` found CVE-2025-69872 (unpatched RCE
  in a `dvc-data` transitive dependency); data is versioned directly in
  Git instead.
- `rag_flagship.corpus` module (manifest and fetcher) and
  `scripts/fetch_corpus.py`; the full 16-document corpus (AI Act, GDPR,
  EDPB guidelines, AI Act guidance, GPAI Code of Practice) is fetched into
  `data/raw/`.
- `rag_flagship.ingestion` module (Docling-based regulation and guidance
  parsers) and `scripts/ingest_corpus.py`; every document is parsed into
  cited passages in `data/processed/*.jsonl` (1597 passages total).
- `rag_flagship.chunking` module: recursive, semantic, and parent-child
  strategies, verified against the real Ollama `bge-m3` model.
- `rag_flagship.embeddings` module (typed Ollama `bge-m3` factory) and
  `rag_flagship.indexing` module (Qdrant hybrid dense + BM25 store, RRF
  queries); `docker-compose.yml` for local Qdrant; `scripts/build_index.py`.
  Full corpus indexed and verified with manual EN/FR hybrid queries.
- `rag_flagship.golden` module and `data/golden/qa_v1.jsonl`: 62
  hand-curated question/answer pairs (factual, multi-hop,
  out-of-corpus; English and French, including a cross-lingual subset),
  each verified against the real processed corpus.
- `.github/workflows/ci.yml` (ruff, mypy, pytest across a Python
  3.11/3.12/3.13 matrix) and `security.yml` (pip-audit, weekly-scheduled;
  gitleaks), every third-party action pinned to a commit SHA.
- Internal engineering documentation (system map, dependency graph,
  data flow, interfaces, usage guide) and the stage 1 audit (65 tests,
  coverage, dependency/security review, SBOM snapshot with 206
  components); README filled in with a verified quick start and
  architecture diagram.

### Fixed

- The first real push of the Semaine 3 changes failed CI twice:
  `pip-audit` fails the build on any finding by default, including two
  already-reviewed, accepted vulnerabilities with no fix available;
  fixed with `--ignore-vuln`. Separately, the CI faithfulness gate's
  original judge model (`qwen2.5:1.5b`) could not reliably produce
  RAGAS's structured metric output on the GitHub-hosted runner's
  CPU-only Ollama build, despite working reliably in local testing on
  a GPU-accelerated machine. After testing five real candidates against
  the gate's exact questions, the CI judge settled on `ministral-3:3b`
  (rejecting `qwen2.5:3b` for one anomalous scoring result, `qwen3.5:4b`
  for unusably slow "thinking mode" generations, `gemma4:E4B` for a
  9.6 GB disk footprint, and `SmolLM3-3B` for not being in Ollama's
  official library).
- RAGAS's own `max_tokens` default (1024) was too small for structured
  metric output against real, multi-sentence generated answers; raised
  to 4096. Separately, Ollama's OpenAI-compatible endpoint was found to
  silently ignore per-request context-window overrides (a genuine
  platform limitation, not a code bug); fixed operationally by
  requiring Ollama to be started with `OLLAMA_CONTEXT_LENGTH=16384`.
- A `ragas`/`langchain-community` import incompatibility: installing
  latest `langchain-community` broke `import ragas` outright (a known,
  unresolved upstream issue). Fixed by pinning
  `langchain-community>=0.4.1,<0.4.2`.
- `ci.yml`/`security.yml` triggered on `push.branches: [master]`, a
  stale reference from before the branch was renamed to `main`; CI had
  never actually run on any push as a result. Fixed to `[main]`.
- Once CI actually ran, the Python 3.13 job failed: `mypy`'s pinned
  `python_version = "3.11"` rejected numpy 2.5.1's stub syntax (numpy
  2.5.1 is only resolved under a 3.13 interpreter). Fixed by letting
  mypy auto-detect its target version from whichever interpreter runs
  it, matching the CI matrix instead of one static value.
- A silent, incorrect reranking bug: `sentence-transformers` would
  auto-select Apple's MPS GPU backend for the reranker, which under
  memory contention with Ollama's own models returned identical scores
  for genuinely different passages without raising any exception.
  Fixed by forcing CPU inference for the reranker; a regression test
  now guards against this recurring.
- ADR-0001 addendum: `eur-lex.europa.eu`'s own pages block plain HTTP
  clients behind an AWS WAF challenge; the four core regulation documents
  are now fetched from the Publications Office's Cellar endpoint instead.

### Removed

- `dvc` dependency, see ADR-0003.
