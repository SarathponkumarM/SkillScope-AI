# SkillScope AI v4 — Complete Candidate Flow and Evaluation-Hardened RAG

This executable Group 83 prototype runs locally without a paid API. Version 4 completes the agreed candidate flow and adds verified evidence anchoring, off-topic answer protection, expanded hallucination/scoring metrics, configurable quality gates and automated GitHub tests.

## Candidate experience

- The director generates a signed link containing the anonymous participant ID, assigned role and expiry.
- The employee does not select a role or question set.
- Every ten-question set contains seven written questions and three coding/technical quizzes.
- Every question can be skipped; a skipped question receives 0%.
- The Begin button requests browser full screen. Browsers require a user gesture and cannot be forcibly locked in full screen; the interface warns and offers a return button if it is exited.
- After submission, full screen closes and the detailed result opens in a new tab.
- The result shows per-question accuracy, expected evidence, missing concepts and approved learning links.

## Marking policy

- Written responses are marked by required technical concepts, not grammar.
- Simple word variants such as `identify`, `identifies` and `identified` are normalised before matching.
- Quiz answers receive 100% when correct and 0% when incorrect or skipped.
- A question reaches competency at 80% accuracy.
- The final result is the arithmetic mean of all ten question accuracies, including skipped questions as zero.
- The deterministic percentage is the official prototype score. Ollama produces developmental explanations for the weakest written answers and does not control the final mark.

## What works now

- Loads 120 role-tagged technical and OTT knowledge chunks.
- Anchors every curated question to its verified authoring evidence and retrieves complementary role-aware TF-IDF passages.
- Compares global retrieval with role-aware retrieval.
- Measures Hit@1, Hit@3, Hit@5, MRR@5 and unanswered rate.
- Builds five deterministic sets for each of five roles (25 sets total).
- Assigns the same participant to the same set after refresh/resume.
- Demonstrates transparent concept-based draft scoring and evidence citations.
- Implements the full RAG code boundary: role-aware retrieval, prompt augmentation,
  Ollama generation, structured-output validation and citation checking.
- Runs 23 automated tests plus the retrieval benchmark on GitHub Actions.

## Run it

Requires Python 3 plus `scikit-learn`.

```bash
python -m pip install -r requirements.txt
python3 evaluate_retrieval.py
python3 build_question_sets.py
python3 -m unittest -v test_prototype.py test_rag_pipeline.py test_v3_features.py
python3 demo.py
python3 web_app.py
```

On Windows, `python` can be used instead of `python3` for every command.

After starting `web_app.py`, the root page explains how to create an invitation. Candidates use their signed invitation URL to open the assigned mixed-format assessment.

For v3, create a candidate link first:

    python web_app.py

In another terminal:

    python create_invitation.py --participant EMP-001 --role qa_test_engineer

Open the printed URL. For access from another machine, set `--base-url` to the permitted hosted or network URL. Set a private secret before starting the web app or generating links:

    $env:SKILLSCOPE_INVITATION_SECRET="use-a-private-random-value-of-at-least-32-characters"

Use the same environment value in both terminals. The repository intentionally contains no default signing secret.

## Important design note

Each set contains ten non-repeated questions. The starter bank has only 14–23 questions per role, so questions can overlap between different sets. Fully unique production sets require expanding the reviewed question bank or adding controlled, validated variants.

The official scorer is intentionally transparent and deterministic. Ollama supplies bounded developmental feedback only. Final performance claims still require the team to label a held-out employee-answer set and compare the generated feedback with at least two technical reviewers.

## Mentor update summary

> SkillScope now implements the complete agreed prototype journey: signed role-specific invitation links, 25 balanced sets, seven written questions plus three quizzes, skip handling, grammar-independent concept accuracy, an 80% competency threshold, a detailed result tab and evidence-grounded learning recommendations. The assessment RAG path guarantees the verified authoring passage at rank one and adds complementary role-aware TF-IDF evidence. Twenty-three automated tests pass. The remaining research activity is human labelling and reporting real-world grounding and scoring-agreement results; example labels are never presented as final performance.

## Ollama RAG evaluation

Install Ollama separately and pull the configured local model:

    ollama pull llama3.2:3b
    ollama serve

In another terminal, run:

    python rag_demo.py --question-id Q_QA_T_01 --answer "I would verify the expected response and test the API behaviour."

The evaluator retrieves five role-specific passages, builds an augmented prompt
containing the question, employee answer, rubric and evidence, calls Ollama, and
rejects citations that were not retrieved. If Ollama is unavailable, the normal
command fails clearly. The optional --fallback flag returns an explicitly
labelled retrieval-only result; it never presents that fallback as RAG.

Run every automated check:

    python -m unittest -v test_prototype.py test_rag_pipeline.py test_v3_features.py

## Hallucination controls

The model receives only the selected role, question, required concepts and retrieved approved evidence. Each curated question is anchored to its verified authoring chunk, while role-aware TF-IDF retrieves complementary passages. This prevents a lexical retrieval miss from removing the approved ground truth. Employee text is explicitly untrusted, temperature is zero, generated citation IDs are rejected unless they were actually retrieved, scores are bounded, failures are labelled as retrieval-only fallback, approved source URLs are supplied by the dataset, and every result requires human review. Expected answers are copied from the curated evidence or fixed quiz explanation rather than generated by Ollama. An off-topic answer with zero approved-concept coverage is deterministically capped at 20% of the question maximum for developmental feedback; the official assessment percentage remains the deterministic concept/quiz score.

The faculty-facing evaluation metrics are:

- Retrieval: Hit@1, Hit@3, Hit@5, MRR@5 and miss/unanswered rate.
- Grounding: citation precision, citation recall, claim groundedness and hallucination rate.
- Scoring: mean absolute error against human scores and 80% competency agreement.
- Reliability: JSON-valid output rate, fallback rate and repeated-run consistency.

Use human-labelled cases with:

    python evaluate_rag_quality.py data/rag_quality_cases.example.jsonl

To save the report and enforce explicit release gates:

```powershell
python evaluate_rag_quality.py data/rag_quality_cases.example.jsonl `
  --output results/rag_quality_metrics.json `
  --max-hallucination-rate 0.10 `
  --min-citation-precision 0.95 `
  --min-competency-agreement 0.80
```

The command exits with status `2` when a supplied quality gate fails. GitHub Actions also runs all tests and the retrieval benchmark on every push and pull request.

The included file demonstrates the calculation only. Final reported hallucination and scoring metrics must use independently human-labelled project cases.
