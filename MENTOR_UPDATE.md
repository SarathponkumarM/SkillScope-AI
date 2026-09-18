# SkillScope AI — Weekly Mentor Update

## What we completed

We moved from the project proposal into a working baseline prototype. It now:

- loads the 120-chunk knowledge base covering five technical roles with OTT content;
- creates five assessment sets per role, giving 25 sets in total;
- assigns a participant to a stable set using an anonymous participant ID;
- accepts written answers through a local browser interface;
- retrieves role-relevant technical evidence;
- produces transparent draft scores and learning gaps for human review; and
- anchors each assessment question to the approved passage used to author it;
- validates generated citation IDs and caps off-topic developmental scores;
- calculates hallucination, citation and scoring-agreement metrics with optional release gates;
- passes 23 automated tests in local verification; and
- runs locally without a paid API.

## First measured result

| Retrieval configuration | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Miss rate @5 |
|---|---:|---:|---:|---:|---:|
| Global TF-IDF | 42.17% | 50.60% | 53.01% | 46.47% | 46.99% |
| Role-aware TF-IDF | 44.58% | 51.81% | 61.45% | 49.92% | 38.55% |
| Assessment RAG (verified anchor + role TF-IDF) | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

Filtering by the employee's role improved baseline Hit@5 by 8.44 percentage points. For the real assessment path, the approved authoring passage is always placed first and TF-IDF supplies additional passages. The 100% figure therefore measures evidence availability for the 83 curated questions; it is not a claim that the language model is always correct.

## What to demonstrate

1. Start the browser application.
2. Generate a signed `QA and Testing` invitation for anonymous ID `DEMO-001`.
3. Open that link and show that the candidate cannot choose another role or set.
4. Accept the consent notice and enter full screen.
5. Answer one written question, complete one quiz and skip one question.
6. Complete the questions and show the evidence-linked draft result.
7. Explain that a technical director reviews the result before it becomes final.

## What we learned

- Role metadata improves retrieval accuracy.
- Keyword retrieval struggles when the question and supporting passage use different wording.
- The current 83-question bank can create 25 ten-question sets without repetition inside one set, but some questions overlap across different sets.
- A deterministic participant-to-set assignment prevents refreshes from changing the test.

## Remaining research validation

Two technical reviewers must label a held-out set of real or consented pilot answers. Run `evaluate_rag_quality.py` on those labels and report MAE, RMSE, within-ten-point rate, competency agreement/F1, citation precision/recall, groundedness and hallucination rate. The three-row example file demonstrates the calculator only and must not be reported as final model performance.

## Short spoken update

> The agreed SkillScope prototype flow is now implemented. A candidate receives a signed role-specific link and completes seven written questions and three quizzes, with a skip option. Official marks use deterministic concept or quiz accuracy, not the LLM. The RAG layer always receives the verified source passage for the question plus complementary role evidence, then Ollama produces developmental feedback whose citations are validated. Twenty-three automated tests pass. We can demonstrate the software now; the remaining research task is to collect human labels and report honest grounding, hallucination and scoring-agreement results.
