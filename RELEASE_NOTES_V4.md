# SkillScope AI Version 4

## Completed prototype scope

- Signed, expiring, role-specific candidate invitation links.
- No signing credential is stored in source code; a private 32+ character environment secret is required.
- No candidate-side role or question-set selection.
- Five roles, five sets per role and ten questions per set.
- Seven written questions and three coding/technical quizzes in every set.
- Skip handling with skipped questions scored as zero.
- Grammar-independent required-concept scoring.
- 80% question competency threshold and mean accuracy across all ten questions.
- Full-screen request during the assessment and a separate result tab after submission.
- Per-question expected evidence, missing concepts, approved learning links and RAG feedback.
- Local Ollama integration with a safe retrieval-only fallback.

## RAG and hallucination hardening

- Every approved assessment question now anchors retrieval to its verified authoring chunk.
- Role-aware TF-IDF fills the remaining evidence positions.
- Generated citation IDs must belong to the supplied evidence.
- Off-topic answers with zero approved-concept coverage cannot receive more than 20% of the maximum developmental model score.
- The official assessment score remains deterministic and separate from model feedback.
- Quality evaluation now reports MAE, RMSE, within-ten-point rate, competency precision/recall/F1, competency agreement, citation precision/recall, groundedness and hallucination rate.
- Optional quality thresholds fail with a non-zero process status for repeatable release checks.

## Verification

- 23 automated unit/integration tests pass.
- The assessment evidence path covers all 83 curated written questions with the verified authoring chunk at rank one: Hit@1 = 1.0, MRR@5 = 1.0 and unanswered rate = 0.0.
- GitHub Actions runs the full test suite and retrieval benchmark on pushes to `main` and on pull requests.

The assessment retrieval score above proves availability of the curated ground-truth passage. It does not prove that every generated sentence is correct. Final model-quality results require a separately human-labelled held-out answer set.
