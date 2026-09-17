"""Core retrieval, set allocation, and rubric scoring for SkillScope AI."""

from __future__ import annotations

import hashlib
import hmac
import json
import base64
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "source"
COMPETENCY_THRESHOLD = 80.0


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9+#.]+", " ", text.lower()).strip()


def _stem(token: str) -> str:
    """Small deterministic stemmer for rubric matching, not grammar marking."""
    token = normalise(token)
    for suffix in ("ization", "isation", "ational", "fulness", "iveness", "ments", "ment", "ingly", "edly", "ing", "ied", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            if suffix in {"ied", "ies"}:
                return token[:-len(suffix)] + "y"
            return token[:-len(suffix)]
    return token


def concept_tokens(text: str) -> set[str]:
    return {_stem(token.strip(".")) for token in normalise(text).split() if token.strip(".")}


def concept_is_present(concept: str, answer: str) -> bool:
    """Match technical keywords and simple word variants; ignore grammar."""
    expected = concept_tokens(concept)
    actual = concept_tokens(answer)
    return bool(expected) and expected.issubset(actual)


@dataclass
class RetrievalResult:
    chunk: dict
    score: float


class RoleAwareRetriever:
    """TF-IDF baseline with optional role filtering."""

    def __init__(self, chunks: Iterable[dict]):
        self.chunks = list(chunks)
        corpus = [
            " ".join(
                [row.get("skill", ""), row.get("level", ""), row.get("text", "")]
            )
            for row in self.chunks
        ]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, role: str | None = None, top_k: int = 5) -> list[RetrievalResult]:
        candidates = [i for i, row in enumerate(self.chunks) if role is None or row["role"] == role]
        if not candidates:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix[candidates]).ravel()
        ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1], self.chunks[item[0]]["chunk_id"]))
        return [RetrievalResult(self.chunks[i], float(score)) for i, score in ranked[:top_k]]


def build_assessment_sets(
    questions: list[dict],
    quizzes: list[dict] | None = None,
    questions_per_set: int = 10,
    quizzes_per_set: int = 3,
) -> dict[str, list[dict]]:
    """Create five deterministic, balanced set blueprints per role.

    Questions never repeat within one set. Because the starter bank has fewer
    than 50 questions per role, base questions can appear in different sets.
    """
    grouped: dict[str, list[dict]] = {}
    for question in questions:
        grouped.setdefault(question["role"], []).append(question)

    output: dict[str, list[dict]] = {}
    for role, role_questions in sorted(grouped.items()):
        ordered = sorted(role_questions, key=lambda row: row["question_id"])
        role_quizzes = sorted(
            [row for row in (quizzes or []) if row["role"] == role],
            key=lambda row: row["question_id"],
        )
        written_count = questions_per_set - (quizzes_per_set if role_quizzes else 0)
        stride = max(1, len(ordered) // 5)
        prefix = {
            "backend_software_engineer": "B",
            "frontend_ui_developer": "U",
            "devops_sre_engineer": "D",
            "qa_test_engineer": "Q",
            "data_engineer": "E",
        }[role]
        sets = []
        for index in range(5):
            start = (index * stride) % len(ordered)
            selected = []
            for offset in range(written_count):
                row = dict(ordered[(start + offset) % len(ordered)])
                row["question_type"] = "written"
                selected.append(row)
            for offset in range(quizzes_per_set if role_quizzes else 0):
                row = dict(role_quizzes[(index + offset) % len(role_quizzes)])
                row["question_type"] = "quiz"
                selected.append(row)
            sets.append({"set_id": f"{prefix}{index + 1}", "role": role, "questions": selected})
        output[role] = sets
    return output


def assign_set(participant_id: str, role: str, sets: dict[str, list[dict]]) -> dict:
    digest = hashlib.sha256(f"{participant_id}:{role}".encode()).hexdigest()
    return sets[role][int(digest[:8], 16) % len(sets[role])]


def score_answer(answer: str, question: dict) -> dict:
    """Grammar-independent concept accuracy for a written answer."""
    concepts = [item.strip() for item in question["required_concepts"].split("|") if item.strip()]
    found = [concept for concept in concepts if concept_is_present(concept, answer)]
    missing = [concept for concept in concepts if concept not in found]
    ratio = len(found) / len(concepts) if concepts else 0.0
    accuracy = round(100 * ratio, 1)
    return {
        "score": round(question.get("max_score", 10) * ratio, 1),
        "max_score": question.get("max_score", 10),
        "accuracy": accuracy,
        "competent": accuracy >= COMPETENCY_THRESHOLD,
        "matched_concepts": found,
        "missing_concepts": missing,
        "gap": "none" if accuracy >= COMPETENCY_THRESHOLD else "development" if accuracy >= 40 else "priority",
    }


def score_quiz(selected_index: int | None, question: dict) -> dict:
    correct = selected_index is not None and int(selected_index) == int(question["correct_index"])
    accuracy = 100.0 if correct else 0.0
    return {
        "score": question.get("max_score", 10) if correct else 0.0,
        "max_score": question.get("max_score", 10),
        "accuracy": accuracy,
        "competent": correct,
        "matched_concepts": [question.get("learning_topic", "quiz concept")] if correct else [],
        "missing_concepts": [] if correct else [question.get("learning_topic", "quiz concept")],
        "gap": "none" if correct else "priority",
    }


def score_response(response: dict, question: dict) -> dict:
    if response.get("skipped"):
        return {
            "score": 0.0, "max_score": question.get("max_score", 10), "accuracy": 0.0,
            "competent": False, "matched_concepts": [],
            "missing_concepts": [x.strip() for x in question.get("required_concepts", question.get("learning_topic", "")).split("|") if x.strip()],
            "gap": "priority", "skipped": True,
        }
    if question.get("question_type") == "quiz":
        result = score_quiz(response.get("selected_index"), question)
    else:
        result = score_answer(str(response.get("answer", "")), question)
    result["skipped"] = False
    return result


def _invitation_secret() -> bytes:
    return os.getenv("SKILLSCOPE_INVITATION_SECRET", "skillscope-local-demo-change-me").encode()


def create_invitation_token(participant_id: str, role: str, expires_at: int | None = None) -> str:
    payload = {"participant_id": participant_id, "role": role, "expires_at": expires_at or int(time.time()) + 7 * 86400}
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).rstrip(b"=")
    signature = hmac.new(_invitation_secret(), encoded, hashlib.sha256).digest()
    return f"{encoded.decode()}.{base64.urlsafe_b64encode(signature).rstrip(b'=').decode()}"


def verify_invitation_token(token: str) -> dict:
    try:
        encoded_text, signature_text = token.split(".", 1)
        encoded = encoded_text.encode()
        supplied = base64.urlsafe_b64decode(signature_text + "=" * (-len(signature_text) % 4))
        expected = hmac.new(_invitation_secret(), encoded, hashlib.sha256).digest()
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("Invalid invitation signature.")
        payload = json.loads(base64.urlsafe_b64decode(encoded_text + "=" * (-len(encoded_text) % 4)))
        if int(payload["expires_at"]) < int(time.time()):
            raise ValueError("This invitation has expired.")
        return payload
    except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ValueError) and str(exc) in {"Invalid invitation signature.", "This invitation has expired."}:
            raise
        raise ValueError("Invalid invitation token.") from exc


def load_quizzes() -> list[dict]:
    return load_jsonl(DATA_DIR / "coding_quizzes.jsonl")


def load_project_data() -> tuple[list[dict], list[dict], list[dict]]:
    return (
        load_jsonl(DATA_DIR / "knowledge_base.jsonl"),
        load_jsonl(DATA_DIR / "assessment_questions.jsonl"),
        load_jsonl(DATA_DIR / "evaluation_cases.jsonl"),
    )
