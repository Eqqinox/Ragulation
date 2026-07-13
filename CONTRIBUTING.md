# Contributing

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) for dependency and environment management
- Docker (for the local Qdrant instance)
- [Ollama](https://ollama.com/) (for local embedding and generation models)

## Setup

```bash
git clone <repository-url>
cd ragulation
uv sync
```

## Development workflow

1. Create a branch for your change.
2. Write or update the code and its tests in the same commit.
3. Run the quality gate locally before opening a pull request:

   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run mypy src
   uv run pytest -q
   uv run pip-audit
   ```

4. Use [Conventional Commits](https://www.conventionalcommits.org/) messages
   and update `CHANGELOG.md` (Keep a Changelog format) for user-visible changes.

## Testing standard

- Unit tests are deterministic: no live network calls, no live LLM calls.
  Tests that require Ollama, Qdrant, or network access are marked
  `@pytest.mark.integration` and are skipped by default (`pytest -m "not integration"`
  is the default `testpaths` behavior; run `pytest -m integration` explicitly
  to exercise them against local services).
- Every bug fix ships with a regression test that fails before the fix and
  passes after.
