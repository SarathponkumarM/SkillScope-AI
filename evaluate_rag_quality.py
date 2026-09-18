"""Calculate human-labelled RAG hallucination and scoring metrics.

Input JSONL fields per case:
human_score, model_score, cited_chunk_ids, allowed_chunk_ids,
claim_support (array of booleans), expected_citation_ids (optional).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


REQUIRED_FIELDS = {
    "human_score",
    "model_score",
    "cited_chunk_ids",
    "allowed_chunk_ids",
    "claim_support",
}


def validate_rows(rows: list[dict]) -> None:
    """Reject incomplete labels so a misleading report is never produced."""
    if not rows:
        raise ValueError("At least one labelled evaluation case is required.")
    for index, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            raise ValueError(f"Case {index} is missing required fields: {missing}")
        for field in ("human_score", "model_score"):
            try:
                value = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Case {index} has a non-numeric {field}.") from exc
            if not 0 <= value <= 100:
                raise ValueError(f"Case {index} {field} must be between 0 and 100.")
        for field in ("cited_chunk_ids", "allowed_chunk_ids", "claim_support"):
            if not isinstance(row[field], list):
                raise ValueError(f"Case {index} field {field} must be a list.")


def calculate_metrics(rows: list[dict], competency_threshold: float = 80.0) -> dict:
    validate_rows(rows)
    errors, agreements = [], []
    squared_errors = []
    within_ten = []
    true_positive = false_positive = false_negative = 0
    valid_citations = total_citations = expected_found = expected_total = 0
    supported_claims = total_claims = 0
    for row in rows:
        human, model = float(row["human_score"]), float(row["model_score"])
        errors.append(abs(human - model))
        squared_errors.append((human - model) ** 2)
        within_ten.append(abs(human - model) <= 10)
        human_positive = human >= competency_threshold
        model_positive = model >= competency_threshold
        agreements.append(human_positive == model_positive)
        true_positive += int(human_positive and model_positive)
        false_positive += int(not human_positive and model_positive)
        false_negative += int(human_positive and not model_positive)
        cited = set(row.get("cited_chunk_ids", []))
        allowed = set(row.get("allowed_chunk_ids", []))
        expected = set(row.get("expected_citation_ids", []))
        valid_citations += len(cited & allowed)
        total_citations += len(cited)
        expected_found += len(cited & expected)
        expected_total += len(expected)
        support = [bool(x) for x in row.get("claim_support", [])]
        supported_claims += sum(support)
        total_claims += len(support)
    groundedness = supported_claims / total_claims if total_claims else 0.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "cases": len(rows),
        "score_mae": round(sum(errors) / len(errors), 4),
        "score_rmse": round(math.sqrt(sum(squared_errors) / len(squared_errors)), 4),
        "scores_within_10_points": round(sum(within_ten) / len(within_ten), 4),
        "competency_agreement": round(sum(agreements) / len(agreements), 4),
        "competency_precision": round(precision, 4),
        "competency_recall": round(recall, 4),
        "competency_f1": round(f1, 4),
        "citation_precision": round(valid_citations / total_citations, 4) if total_citations else 0.0,
        "citation_recall": round(expected_found / expected_total, 4) if expected_total else 0.0,
        "claim_groundedness": round(groundedness, 4),
        "hallucination_rate": round(1.0 - groundedness, 4),
        "labelled_claims": total_claims,
        "model_citations": total_citations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Human-labelled JSONL file")
    parser.add_argument("--output", help="Optional JSON output path")
    parser.add_argument("--max-hallucination-rate", type=float)
    parser.add_argument("--min-citation-precision", type=float)
    parser.add_argument("--min-competency-agreement", type=float)
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.path).read_text(encoding="utf-8").splitlines() if line.strip()]
    metrics = calculate_metrics(rows)
    gates = {
        "hallucination_rate": args.max_hallucination_rate is None
        or metrics["hallucination_rate"] <= args.max_hallucination_rate,
        "citation_precision": args.min_citation_precision is None
        or metrics["citation_precision"] >= args.min_citation_precision,
        "competency_agreement": args.min_competency_agreement is None
        or metrics["competency_agreement"] >= args.min_competency_agreement,
    }
    report = {**metrics, "quality_gates": gates, "all_quality_gates_passed": all(gates.values())}
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["all_quality_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
