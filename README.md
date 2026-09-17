# SkillScope AI v3 — Candidate Link and RAG Learning Feedback

This executable Group 83 prototype runs locally without a paid API. Version 3 adds signed candidate links, mixed written and coding-quiz sets, skip handling, grammar-independent concept accuracy, detailed employee results and bounded RAG learning feedback.

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
- Retrieves evidence with a TF-IDF baseline.
- Compares global retrieval with role-aware retrieval.
- Measures Hit@1, Hit@3, Hit@5, MRR@5 and unanswered rate.
- Builds five deterministic sets for each of five roles (25 sets total).
- Assigns the same participant to the same set after refresh/resume.
- Demonstrates transparent concept-based draft scoring and evidence citations.
- Implements the full RAG code boundary: role-aware retrieval, prompt augmentation,
  Ollama generation, structured-output validation and citation checking.

## Run it

Requires Python 3 plus `scikit-learn`.

```bash
python -m pip install -r requirements.txt
python evaluate_retrieval.py
python build_question_sets.py
python -m unittest -v test_prototype.py test_rag_pipeline.py test_v3_features.py
python web_app.py
```

In another terminal:

```powershell
python create_invitation.py --participant EMP-001 --role qa_test_engineer
```

Open the printed URL. Before shared deployment, set a private invitation secret:

```powershell
$env:SKILLSCOPE_INVITATION_SECRET="replace-with-a-long-random-secret"
```

## Ollama RAG evaluation

Install Ollama separately, then run:

```powershell
ollama pull llama3.2:3b
python rag_demo.py --question-id Q_QA_T_01 --answer "I would verify the expected response and test the API behaviour."
```

The deterministic score remains official. Local Ollama generates bounded developmental feedback for weak written responses.

## Hallucination controls and metrics

The model receives only the selected role, question, required concepts and retrieved approved evidence. Employee text is treated as untrusted. Temperature is zero, unsupported citation IDs are rejected, scores are bounded, approved source links come from the dataset, and all AI feedback is reviewable.

Evaluation covers:

- Retrieval: Hit@1, Hit@3, Hit@5, MRR@5 and miss rate.
- Grounding: citation precision/recall, claim groundedness and hallucination rate.
- Scoring: mean absolute error and 80% competency agreement against human labels.
- Reliability: JSON-valid output, fallback rate and repeated-run consistency.

See `RAG_ARCHITECTURE.md`, `HALLUCINATION_EVALUATION.md` and `RELEASE_NOTES_V3.md` for details.
