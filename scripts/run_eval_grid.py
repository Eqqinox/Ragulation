"""RAGAS comparison grid across chunking strategy, retrieval mode, and
reranker on/off, per projet_1_rag_flagship.md's Semaine 3 scope.

Usage:
    uv run python scripts/run_eval_grid.py grid
    uv run python scripts/run_eval_grid.py full --strategy recursive \
        --retrieval-mode hybrid --use-reranker

Scope note (see ADR-0006 and documentations/audit/stage-0003-evaluation.md):
RAGAS's four metrics (faithfulness, answer relevancy, context precision,
context recall) are answer-quality metrics; they have no meaningful
application to a question the pipeline should refuse. Both commands
below therefore run only the golden dataset's `factual` and `multi_hop`
categories (56 of the 62 pairs), not the 6 `out_of_corpus` traps, which
are already covered by the Semaine 2 refusal-threshold calibration
(scripts/calibrate_refusal_threshold.py) and its own tests.

Requires a local Qdrant instance with all three strategies indexed
(docker compose up -d; scripts/build_index.py --strategy ... for each),
and a local Ollama server with mistral-small3.2, bge-m3 pulled.
"""

from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import typer
from qdrant_client import QdrantClient

from rag_flagship.embeddings.dense import build_dense_embedding_model
from rag_flagship.evaluation.dataset import build_eval_sample
from rag_flagship.evaluation.judge import (
    build_judge_client,
    build_judge_embeddings,
    build_judge_llm,
)
from rag_flagship.evaluation.pipeline import EvalResult, run_ragas_eval
from rag_flagship.generation.llm import build_generation_model
from rag_flagship.generation.prompt import build_messages
from rag_flagship.golden.models import GoldenQAPair
from rag_flagship.indexing.pipeline import dense_query, hybrid_query
from rag_flagship.indexing.settings import QdrantSettings
from rag_flagship.indexing.store import build_vector_store, collection_name_for_strategy
from rag_flagship.reranking.cross_encoder import build_reranker
from rag_flagship.reranking.pipeline import rerank

app = typer.Typer(add_completion=False)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "qa_v1.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "documentations" / "audit"

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5

CHUNKING_STRATEGIES = ("recursive", "semantic", "parent_child")
RETRIEVAL_MODES = ("dense", "hybrid")
RERANKER_OPTIONS = (False, True)

# Fixed, stratified subsample for the 12-config grid: 12 questions (5
# multi_hop, 7 factual spread across every document/language
# combination in the golden set), sized down from an original 24 after
# measuring ~243s/sample on the real local stack (a 24-question x
# 12-config grid would have taken ~19 hours; 12 questions brings this
# to ~10 hours). Fixed and named, not randomly re-sampled each run, so
# the 12 results stay directly comparable across runs. See ADR-0006.
GRID_SUBSAMPLE_QA_IDS = frozenset(
    {
        "factual_en_gdpr_001",
        "factual_en_gdpr_005",
        "factual_en_aiact_001",
        "factual_en_aiact_005",
        "factual_en_guidance_001",
        "factual_fr_gdpr_001",
        "factual_fr_aiact_001",
        "multihop_001",
        "multihop_003",
        "multihop_005",
        "multihop_007",
        "multihop_009",
    }
)


def load_golden_pairs(path: Path) -> list[GoldenQAPair]:
    with path.open() as f:
        return [GoldenQAPair.model_validate_json(line) for line in f if line.strip()]


def answerable_pairs(golden_pairs: list[GoldenQAPair]) -> list[GoldenQAPair]:
    """factual and multi_hop only; out_of_corpus is excluded, see the
    module docstring's scope note."""
    return [pair for pair in golden_pairs if pair.category != "out_of_corpus"]


class SharedResources:
    def __init__(self) -> None:
        self.qdrant_client = QdrantClient(
            url=QdrantSettings().url, api_key=QdrantSettings().api_key
        )
        self.embed_model = build_dense_embedding_model()
        self.reranker = build_reranker()
        self.llm = build_generation_model()
        judge_client = build_judge_client()
        self.judge_llm = build_judge_llm(judge_client)
        self.judge_embeddings = build_judge_embeddings(judge_client)


def run_one_config(
    golden_pairs: list[GoldenQAPair],
    strategy: str,
    retrieval_mode: str,
    use_reranker: bool,
    resources: SharedResources,
) -> list[EvalResult]:
    collection_name = collection_name_for_strategy(strategy)
    vector_store = build_vector_store(collection_name, client=resources.qdrant_client)
    query_fn = hybrid_query if retrieval_mode == "hybrid" else dense_query

    samples = []
    for golden in golden_pairs:
        candidates = query_fn(
            vector_store, resources.embed_model, golden.question, top_k=RETRIEVAL_TOP_K
        )
        if not candidates:
            continue
        ranked = (
            rerank(golden.question, candidates, resources.reranker, RERANK_TOP_K)
            if use_reranker
            else list(candidates)[:RERANK_TOP_K]
        )

        messages = build_messages(golden.question, golden.question_language, ranked)
        response = resources.llm.chat(messages)
        answer_text = response.message.content or ""

        contexts = [node.node.get_content() for node in ranked]
        samples.append(build_eval_sample(golden, answer_text, contexts))

    if not samples:
        return []
    return run_ragas_eval(samples, resources.judge_llm, resources.judge_embeddings)


def summarize(
    strategy: str, retrieval_mode: str, use_reranker: bool, results: list[EvalResult]
) -> dict:
    n = len(results)
    if n == 0:
        return {
            "strategy": strategy,
            "retrieval_mode": retrieval_mode,
            "use_reranker": use_reranker,
            "n_samples": 0,
        }
    return {
        "strategy": strategy,
        "retrieval_mode": retrieval_mode,
        "use_reranker": use_reranker,
        "n_samples": n,
        "mean_faithfulness": sum(r.faithfulness for r in results) / n,
        "mean_answer_relevancy": sum(r.answer_relevancy for r in results) / n,
        "mean_context_precision": sum(r.context_precision for r in results) / n,
        "mean_context_recall": sum(r.context_recall for r in results) / n,
    }


def write_markdown_table(summaries: list[dict], path: Path) -> None:
    ranked = sorted(summaries, key=lambda s: s.get("mean_faithfulness", 0.0), reverse=True)
    lines = [
        "| Strategy | Retrieval | Reranker | N | Faithfulness | Answer relevancy |"
        " Context precision | Context recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in ranked:
        if s["n_samples"] == 0:
            lines.append(
                f"| {s['strategy']} | {s['retrieval_mode']} | {s['use_reranker']} "
                "| 0 | - | - | - | - |"
            )
            continue
        lines.append(
            f"| {s['strategy']} | {s['retrieval_mode']} | {s['use_reranker']} | {s['n_samples']} | "
            f"{s['mean_faithfulness']:.3f} | {s['mean_answer_relevancy']:.3f} | "
            f"{s['mean_context_precision']:.3f} | {s['mean_context_recall']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


@app.command()
def grid(
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    limit_questions: int = typer.Option(
        0, help="If > 0, only use this many subsample questions (for a quick smoke test)."
    ),
    limit_configs: int = typer.Option(
        0, help="If > 0, only run this many of the 12 configs (for a quick smoke test)."
    ),
) -> None:
    """Run all 12 configs against the fixed stratified subsample."""
    golden_pairs = load_golden_pairs(golden_path)
    subsample = [p for p in golden_pairs if p.qa_id in GRID_SUBSAMPLE_QA_IDS]
    if limit_questions > 0:
        subsample = subsample[:limit_questions]

    configs = list(product(CHUNKING_STRATEGIES, RETRIEVAL_MODES, RERANKER_OPTIONS))
    if limit_configs > 0:
        configs = configs[:limit_configs]

    resources = SharedResources()

    raw: dict[str, list[dict]] = {}
    summaries = []
    for strategy, retrieval_mode, use_reranker in configs:
        config_key = f"{strategy}/{retrieval_mode}/reranker={use_reranker}"
        start = time.monotonic()
        typer.echo(f"running {config_key} ({len(subsample)} questions)...")
        results = run_one_config(subsample, strategy, retrieval_mode, use_reranker, resources)
        elapsed = time.monotonic() - start
        typer.echo(f"  done in {elapsed:.0f}s, {len(results)} samples scored")

        raw[config_key] = [r.model_dump() for r in results]
        summaries.append(summarize(strategy, retrieval_mode, use_reranker, results))

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage-0003-eval-grid-raw.json").write_text(json.dumps(raw, indent=2))
    write_markdown_table(summaries, output_dir / "stage-0003-eval-grid.md")
    typer.echo(json.dumps(summaries, indent=2))


@app.command()
def full(
    strategy: str = typer.Option(...),
    retrieval_mode: str = typer.Option(...),
    use_reranker: bool = typer.Option(...),
    golden_path: Path = DEFAULT_GOLDEN_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Run one winning config against the full answerable set (56
    factual + multi_hop pairs, out_of_corpus excluded, see module
    docstring) to produce the headline README numbers."""
    golden_pairs = load_golden_pairs(golden_path)
    full_set = answerable_pairs(golden_pairs)

    resources = SharedResources()
    start = time.monotonic()
    config_label = f"{strategy}/{retrieval_mode}/reranker={use_reranker}"
    typer.echo(f"running {config_label} ({len(full_set)} questions)...")
    results = run_one_config(full_set, strategy, retrieval_mode, use_reranker, resources)
    elapsed = time.monotonic() - start
    typer.echo(f"done in {elapsed:.0f}s, {len(results)} samples scored")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage-0003-eval-full-raw.json").write_text(
        json.dumps([r.model_dump() for r in results], indent=2)
    )
    summary = summarize(strategy, retrieval_mode, use_reranker, results)
    (output_dir / "stage-0003-eval-full-summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


if __name__ == "__main__":
    app()
