# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

- ADR-0001 addendum: `eur-lex.europa.eu`'s own pages block plain HTTP
  clients behind an AWS WAF challenge; the four core regulation documents
  are now fetched from the Publications Office's Cellar endpoint instead.

### Removed

- `dvc` dependency, see ADR-0003.
