import os
import time
import unittest

from evaluate_rag_quality import calculate_metrics, validate_rows
from rag_pipeline import RAGEvaluator
from skillscope_core import RoleAwareRetriever, load_project_data
from skillscope_core import (
    create_invitation_token, load_quizzes, score_answer, score_response,
    verify_invitation_token,
)


class SkillScopeV3Tests(unittest.TestCase):
    def setUp(self):
        self.previous_secret = os.environ.get("SKILLSCOPE_INVITATION_SECRET")
        os.environ["SKILLSCOPE_INVITATION_SECRET"] = "x" * 32

    def tearDown(self):
        if self.previous_secret is None:
            os.environ.pop("SKILLSCOPE_INVITATION_SECRET", None)
        else:
            os.environ["SKILLSCOPE_INVITATION_SECRET"] = self.previous_secret

    def test_word_variants_are_matched_without_grammar_marking(self):
        question = {"required_concepts": "identify|object", "max_score": 10}
        result = score_answer("The service identifies and identified the object.", question)
        self.assertEqual(result["accuracy"], 100.0)
        self.assertTrue(result["competent"])

    def test_eighty_percent_threshold(self):
        question = {"required_concepts": "one|two|three|four|five", "max_score": 10}
        result = score_answer("one two three four", question)
        self.assertEqual(result["accuracy"], 80.0)
        self.assertTrue(result["competent"])

    def test_skip_scores_zero(self):
        question = {"required_concepts": "actual|expected", "max_score": 10, "question_type": "written"}
        result = score_response({"skipped": True}, question)
        self.assertEqual(result["accuracy"], 0.0)
        self.assertTrue(result["skipped"])

    def test_quiz_exact_scoring(self):
        quiz = dict(load_quizzes()[0]); quiz["question_type"] = "quiz"
        correct = score_response({"selected_index": quiz["correct_index"]}, quiz)
        wrong = score_response({"selected_index": (quiz["correct_index"] + 1) % 4}, quiz)
        self.assertEqual((correct["accuracy"], wrong["accuracy"]), (100.0, 0.0))

    def test_signed_invitation_round_trip_and_tamper_rejection(self):
        token = create_invitation_token("EMP-42", "qa_test_engineer")
        self.assertEqual(verify_invitation_token(token)["participant_id"], "EMP-42")
        payload, signature = token.split(".", 1)
        tampered = payload + "." + ("A" if signature[0] != "A" else "B") + signature[1:]
        with self.assertRaises(ValueError):
            verify_invitation_token(tampered)

    def test_expired_invitation_rejected(self):
        token = create_invitation_token("EMP-42", "qa_test_engineer", int(time.time()) - 1)
        with self.assertRaisesRegex(ValueError, "expired"):
            verify_invitation_token(token)

    def test_short_invitation_secret_is_rejected(self):
        os.environ["SKILLSCOPE_INVITATION_SECRET"] = "short"
        with self.assertRaisesRegex(RuntimeError, "at least 32 characters"):
            create_invitation_token("EMP-42", "qa_test_engineer")

    def test_hallucination_metrics(self):
        metrics = calculate_metrics([{
            "human_score": 80, "model_score": 70,
            "cited_chunk_ids": ["A", "X"], "allowed_chunk_ids": ["A", "B"],
            "expected_citation_ids": ["A"], "claim_support": [True, False],
        }])
        self.assertEqual(metrics["score_mae"], 10.0)
        self.assertEqual(metrics["citation_precision"], 0.5)
        self.assertEqual(metrics["hallucination_rate"], 0.5)
        self.assertEqual(metrics["score_rmse"], 10.0)
        self.assertEqual(metrics["scores_within_10_points"], 1.0)

    def test_quality_labels_are_validated(self):
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            validate_rows([{"human_score": 80, "model_score": 80}])

    def test_out_of_range_quality_score_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 100"):
            validate_rows([{
                "human_score": 101, "model_score": 80,
                "cited_chunk_ids": [], "allowed_chunk_ids": [], "claim_support": [],
            }])

    def test_off_topic_rag_score_is_capped_and_flagged(self):
        chunks, questions, _ = load_project_data()
        question = questions[0]

        class OverGenerousGenerator:
            def generate_json(self, system_prompt, user_prompt):
                return {
                    "score": 10, "strengths": ["none"],
                    "missing_concepts": [], "improvement_areas": [],
                    "evidence_chunk_ids": [], "confidence": "high",
                    "rationale": "Over-generous test output",
                }

        evaluator = RAGEvaluator(RoleAwareRetriever(chunks), generator=OverGenerousGenerator())
        result = evaluator.evaluate(question, "bananas purple bicycle")
        self.assertLessEqual(result["score"], 2.0)
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["official_accuracy"], 0.0)
        self.assertTrue(result["human_review_required"])

    def test_assessment_retrieval_always_contains_verified_reference(self):
        chunks, questions, _ = load_project_data()
        retriever = RoleAwareRetriever(chunks)
        for question in questions:
            results = retriever.retrieve_for_assessment(question, top_k=5)
            self.assertEqual(results[0].chunk["chunk_id"], question["reference_chunk_id"])
            self.assertTrue(all(item.chunk["role"] == question["role"] for item in results))

    def test_missing_assessment_reference_is_rejected(self):
        chunks, questions, _ = load_project_data()
        broken = dict(questions[0]); broken["reference_chunk_id"] = "MISSING"
        with self.assertRaisesRegex(ValueError, "missing evidence chunk"):
            RoleAwareRetriever(chunks).retrieve_for_assessment(broken)


if __name__ == "__main__":
    unittest.main()
