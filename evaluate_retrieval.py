"""Evaluate global and role-aware SkillScope TF-IDF retrieval."""

from __future__ import annotations

import json
from pathlib import Path

from skillscope_core import RoleAwareRetriever, load_project_data


def metrics_for(retriever, questions, role_aware: bool) -> dict:
    ranks = []
    errors = []
    for question in questions:
        results = retriever.retrieve(
            question["question"],
            role=question["role"] if role_aware else None,
            top_k=5,
        )
        ids = [result.chunk["chunk_id"] for result in results]
        expected = question["reference_chunk_id"]
        rank = ids.index(expected) + 1 if expected in ids else None
        ranks.append(rank)
        if rank is None:
            errors.append({"question_id": question["question_id"], "expected": expected, "retrieved": ids})

    count = len(ranks)
    return {
        "questions": count,
        "hit_at_1": round(sum(rank == 1 for rank in ranks) / count, 4),
        "hit_at_3": round(sum(rank is not None and rank <= 3 for rank in ranks) / count, 4),
        "hit_at_5": round(sum(rank is not None and rank <= 5 for rank in ranks) / count, 4),
        "mrr_at_5": round(sum(1 / rank for rank in ranks if rank) / count, 4),
        "unanswered_rate_at_5": round(sum(rank is None for rank in ranks) / count, 4),
        "misses": errors,
    }


def main() -> None:
    chunks, questions, _ = load_project_data()
    retriever = RoleAwareRetriever(chunks)
    report = {
        "dataset": {"knowledge_chunks": len(chunks), "questions": len(questions)},
        "global_tfidf": metrics_for(retriever, questions, False),
        "role_aware_tfidf": metrics_for(retriever, questions, True),
    }
    output = Path(__file__).resolve().parent / "results" / "retrieval_metrics.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "dataset"}, indent=2))


if __name__ == "__main__":
    main()
