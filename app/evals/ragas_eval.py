"""
app/evals/ragas_eval.py — Ragas evaluation runner for the RAG pipeline.

Ragas metrics evaluated:
  - faithfulness:        Is the answer supported by the retrieved context?
                         (LLM extracts claims; checks each against context)
  - answer_relevancy:   Is the answer relevant to the question?
                         (embeds generated questions from answer; sim to original)
  - context_precision:  Are the retrieved chunks actually needed to answer?
                         (how much of the context is relevant)
  - context_recall:     Does the retrieved context cover the ground-truth answer?
                         (how much of the ground truth can be attributed to context)

Run this script to produce evals_results/ragas_results.json:
    uv run python -m app.evals.ragas_eval

Interview notes:
  "How do you evaluate a RAG system without human labels?"
  → Ragas is "reference-free" for faithfulness + answer_relevancy — it uses
    an LLM-as-judge. context_precision/recall need ground-truth answers.
  "What is judge-model bias?"
  → If your eval judge is the same model you're evaluating, it tends to score
    its own outputs highly. Mitigate by using a stronger / different model for judging.
  "How do you prevent eval-set rot?"
  → Treat the golden set as code: review changes in PRs, add new cases when
    bugs are found, track score trends over time (don't just look at latest).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Golden Q&A set ────────────────────────────────────────────────────────────
# Small handcrafted set tied to our 3 KB files.
# Each entry: question the eval runs against, ground_truth for recall/precision.

GOLDEN_QA: list[dict[str, str]] = [
    {
        "question": "How do I invite team members to my Acme account?",
        "ground_truth": (
            "Go to Settings → Team and click Invite Member. "
            "Enter their email address. They will receive an invitation valid for 7 days. "
            "Available roles are Owner, Admin, Member, and Viewer."
        ),
    },
    {
        "question": "What happens after 5 failed login attempts?",
        "ground_truth": (
            "After 5 failed login attempts, your account is locked for 15 minutes. "
            "You can contact support for an immediate unlock."
        ),
    },
    {
        "question": "What is the maximum file upload size?",
        "ground_truth": "The maximum file size per upload is 250 MB.",
    },
    {
        "question": "How much does the Pro plan cost and how many users does it include?",
        "ground_truth": (
            "The Pro plan costs $29 per month and supports up to 15 users with 50 GB of storage."
        ),
    },
    {
        "question": "What is the API rate limit for the Starter plan?",
        "ground_truth": (
            "The Starter plan allows 60 requests per minute and 10,000 requests per day."
        ),
    },
]

RESULTS_PATH = Path(__file__).resolve().parents[2] / "evals_results" / "ragas_results.json"


# ── Runner ────────────────────────────────────────────────────────────────────

def run_ragas_eval(
    golden_qa: list[dict[str, str]] | None = None,
    *,
    use_hybrid: bool = True,
    top_k: int = 5,
    save: bool = True,
) -> dict:
    """
    Run the Ragas evaluation suite over the golden Q&A set.

    Steps
    -----
    1. For each question: run retrieval to get context chunks.
    2. Call the LLM with the RAG answer prompt to generate an answer.
    3. Build a Ragas Dataset (question, answer, contexts, ground_truth).
    4. Run Ragas evaluate() with all four metrics.
    5. Save results to evals_results/ragas_results.json.

    Returns
    -------
    Dict with per-metric scores and per-question breakdown.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    from app.llm import llm
    from app.prompts.loader import load
    from app.rag.retrieve import format_context

    golden_qa = golden_qa or GOLDEN_QA

    if use_hybrid:
        from app.rag.hybrid_retrieve import hybrid_retrieve
        retrieve_fn = lambda q: hybrid_retrieve(q, top_k=top_k, use_rerank=False, use_mmr=False)
    else:
        from app.rag.retrieve import retrieve
        retrieve_fn = lambda q: retrieve(q, top_k=top_k)

    questions, answers, contexts_list, ground_truths = [], [], [], []

    for item in golden_qa:
        q = item["question"]
        gt = item["ground_truth"]

        logger.info("Eval: '%s'", q)
        chunks = retrieve_fn(q)
        context = format_context(chunks)

        if context:
            prompt = load("rag_answer", version=1, company="Acme", context=context, question=q)
            answer, _ = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        else:
            answer = "I could not find relevant information in the knowledge base."

        questions.append(q)
        answers.append(answer)
        contexts_list.append([c.text for c in chunks] if chunks else [""])
        ground_truths.append(gt)

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
    }

    output = {
        "scores": scores,
        "retrieval_mode": "hybrid" if use_hybrid else "dense_only",
        "top_k": top_k,
        "num_questions": len(golden_qa),
        "per_question": [
            {"question": q, "answer": a, "contexts": ctx, "ground_truth": gt}
            for q, a, ctx, gt in zip(questions, answers, contexts_list, ground_truths)
        ],
    }

    if save:
        RESULTS_PATH.parent.mkdir(exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(output, f, indent=2)
        logger.info("Ragas results saved to %s", RESULTS_PATH)

    return output


def load_results() -> dict | None:
    """Load previously saved Ragas results, or None if not found."""
    if not RESULTS_PATH.exists():
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    use_hybrid = "--dense" not in sys.argv
    output = run_ragas_eval(use_hybrid=use_hybrid)

    print("\n── Ragas Evaluation Results ─────────────────────────────────")
    for metric, score in output["scores"].items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {metric:22s} {bar} {score:.4f}")
    print(f"\n  Questions evaluated: {output['num_questions']}")
    print(f"  Retrieval mode: {output['retrieval_mode']}")
    print(f"  Results saved to: {RESULTS_PATH}")
