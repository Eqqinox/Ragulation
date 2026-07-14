"""CI faithfulness gate: a small, fast, single-configuration RAGAS check.

Not the local comparison grid (see scripts/run_eval_grid.py) -- this is
a narrow regression gate meant to run in GitHub Actions on every push
and PR, against a tiny fixed corpus subset (gdpr_en only, ~440 KB) and
3 fixed questions, using a small CI-sized judge model. See ADR-0006 for
why the CI judge differs from the local/headline judge:
mistral-small3.2 (15 GB) does not fit a GitHub-hosted runner's ~14 GB
disk budget, so OLLAMA_JUDGE_MODEL_NAME must be overridden to a small
model in the CI environment (see .github/workflows/eval.yml).

Usage:
    uv run python scripts/run_ci_eval_gate.py

Exits 1 (failing the CI job) if mean faithfulness falls below
FAITHFULNESS_FLOOR, or if no sample could be scored at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

from rag_flagship.chunking.recursive import RecursiveChunker
from rag_flagship.embeddings.dense import build_dense_embedding_model
from rag_flagship.evaluation.dataset import build_eval_sample
from rag_flagship.evaluation.judge import (
    build_judge_client,
    build_judge_embeddings,
    build_judge_llm,
)
from rag_flagship.evaluation.pipeline import run_ragas_eval
from rag_flagship.generation.llm import build_generation_model
from rag_flagship.generation.prompt import build_messages
from rag_flagship.golden.models import GoldenQAPair
from rag_flagship.indexing.pipeline import hybrid_query, index_chunks
from rag_flagship.indexing.store import build_vector_store
from rag_flagship.ingestion.loader import load_processed_passages
from rag_flagship.reranking.cross_encoder import build_reranker
from rag_flagship.reranking.pipeline import rerank

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
GOLDEN_PATH = REPO_ROOT / "data" / "golden" / "qa_v1.jsonl"
CI_COLLECTION = "rag_flagship_ci_gate"

# Fixed: all three are sourced only from gdpr_en, so indexing just that
# one document (not the full 16-document corpus) is sufficient to
# answer them. Kept identical across runs, not randomly re-sampled.
CI_GATE_QA_IDS = ("factual_en_gdpr_001", "factual_en_gdpr_003", "factual_en_gdpr_005")

RETRIEVAL_TOP_K = 20
RERANK_TOP_K = 5

# Provisional: this is the first CI gate ever configured for this
# project, so no real CI baseline exists yet to set a tolerance band
# around (the discipline used everywhere else in this project, e.g.
# the 0.27 refusal threshold in generation.pipeline, ADR-0005). 0.5 is
# a coarse sanity floor meant to catch a gross regression (the pipeline
# stops generating faithful answers at all), not a tuned number.
# Recalibrate once a few real CI runs establish an actual measured
# baseline for this specific small judge model.
FAITHFULNESS_FLOOR = 0.5


def load_golden_pairs(path: Path) -> list[GoldenQAPair]:
    with path.open() as f:
        return [GoldenQAPair.model_validate_json(line) for line in f if line.strip()]


def main() -> int:
    passages = [p for p in load_processed_passages(PROCESSED_DIR) if p.doc_id == "gdpr_en"]

    embed_model = build_dense_embedding_model()
    chunks = RecursiveChunker().chunk(passages)
    vector_store = build_vector_store(CI_COLLECTION)
    index_chunks(chunks, vector_store=vector_store, embed_model=embed_model)

    reranker = build_reranker()
    llm = build_generation_model()

    golden_pairs = [p for p in load_golden_pairs(GOLDEN_PATH) if p.qa_id in CI_GATE_QA_IDS]

    judge_client = build_judge_client()
    judge_llm = build_judge_llm(judge_client)
    judge_embeddings = build_judge_embeddings(judge_client)

    samples = []
    for golden in golden_pairs:
        candidates = hybrid_query(vector_store, embed_model, golden.question, top_k=RETRIEVAL_TOP_K)
        if not candidates:
            continue
        ranked = rerank(golden.question, candidates, reranker, RERANK_TOP_K)
        messages = build_messages(golden.question, golden.question_language, ranked)
        response = llm.chat(messages)
        answer_text = response.message.content or ""
        contexts = [node.node.get_content() for node in ranked]
        samples.append(build_eval_sample(golden, answer_text, contexts))

    if not samples:
        print("No samples were scored (empty retrieval for every question); failing the gate.")
        return 1

    results = run_ragas_eval(samples, judge_llm, judge_embeddings)
    mean_faithfulness = sum(r.faithfulness for r in results) / len(results)

    print(f"mean_faithfulness={mean_faithfulness:.3f} over {len(results)} samples")
    if mean_faithfulness < FAITHFULNESS_FLOOR:
        print(f"FAILED: mean faithfulness below the {FAITHFULNESS_FLOOR} floor")
        return 1

    print("PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
