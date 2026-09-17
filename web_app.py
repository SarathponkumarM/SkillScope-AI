"""SkillScope v3 candidate-link assessment and evidence-grounded results."""

from __future__ import annotations

import json
import os
import uuid
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from rag_pipeline import RAGEvaluator
from skillscope_core import (
    RoleAwareRetriever, assign_set, build_assessment_sets, load_project_data,
    load_quizzes, score_response, verify_invitation_token,
)

CHUNKS, QUESTIONS, _ = load_project_data()
QUIZZES = load_quizzes()
CHUNK_BY_ID = {row["chunk_id"]: row for row in CHUNKS}
RETRIEVER = RoleAwareRetriever(CHUNKS)
RAG = RAGEvaluator(RETRIEVER)
SETS = build_assessment_sets(QUESTIONS, QUIZZES)
RESULTS: dict[str, dict] = {}
RAG_LIMIT = max(0, int(os.getenv("SKILLSCOPE_RAG_MAX_QUESTIONS", "2")))

ROLES = {
    "backend_software_engineer": "Backend Software Engineering",
    "frontend_ui_developer": "Frontend and UI Development",
    "devops_sre_engineer": "DevOps and SRE",
    "qa_test_engineer": "QA and Testing",
    "data_engineer": "Data Engineering",
}

BASE_STYLE = """
:root{--navy:#10263d;--blue:#1677ff;--soft:#eef5ff;--ink:#17202a;--muted:#65758b;--green:#11845b;--red:#b42318;--amber:#b54708}
*{box-sizing:border-box}body{margin:0;background:#f4f7fb;color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif}
header{background:var(--navy);color:white;padding:20px max(5vw,24px)}header b{font-size:22px}header span{opacity:.72;margin-left:10px}
main{max-width:980px;margin:34px auto;padding:0 20px}.card{background:white;border:1px solid #dfe7f1;border-radius:16px;padding:28px;box-shadow:0 8px 28px #24476a12}
h1{font-size:30px;margin:0 0 8px}h2{margin:18px 0 8px}h3{margin:10px 0}.lead,.muted{color:var(--muted)}.hidden{display:none}
.pill{display:inline-block;background:var(--soft);color:#075bbd;border-radius:99px;padding:5px 11px;font-size:13px;font-weight:700}.progress{height:8px;background:#e8edf3;border-radius:99px;overflow:hidden;margin:17px 0}.progress i{display:block;height:100%;background:var(--blue)}
textarea{width:100%;min-height:150px;padding:12px;border:1px solid #b9c7d8;border-radius:9px;font:inherit;resize:vertical}.actions{display:flex;gap:10px;flex-wrap:wrap}
button{background:var(--blue);border:0;border-radius:9px;color:white;padding:12px 19px;font-weight:700;cursor:pointer;margin-top:18px}button.secondary{background:white;color:var(--blue);border:1px solid var(--blue)}button.skip{background:white;color:#6b7280;border:1px solid #9ca3af}
.notice{background:#fff8e8;border:1px solid #f2d58a;padding:12px;border-radius:9px;margin:18px 0}.option{display:block;border:1px solid #d0d9e5;border-radius:10px;padding:12px;margin:10px 0;cursor:pointer}.option:hover{background:#f7faff}.option input{margin-right:10px}
pre{background:#101828;color:#f8fafc;padding:16px;border-radius:10px;overflow:auto}.score{font-size:44px;font-weight:800;color:var(--green)}.result-item{border:1px solid #dfe7f1;border-radius:12px;padding:18px;margin:16px 0}.good{color:var(--green)}.bad{color:var(--red)}.warn{color:var(--amber)}
.evidence{background:#f7f9fc;border-left:4px solid var(--blue);padding:13px 15px;margin:10px 0}a{color:#075bbd}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:650px){.grid{grid-template-columns:1fr}.card{padding:20px}header span{display:block;margin-left:0}}
"""

CANDIDATE_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SkillScope Assessment</title><style>__STYLE__</style></head><body><header><b>SkillScope AI</b><span>Evidence-grounded technical assessment</span></header><main>
<section id="ready" class="card"><span class="pill">Assigned assessment</span><h1>Welcome to your technical assessment</h1><p class="lead">Your role and question set have already been assigned through this private link.</p>
<div class="grid"><div><b>Participant</b><br>__PARTICIPANT__</div><div><b>Technical domain</b><br>__ROLE__</div></div>
<div class="notice">The assessment has ten questions: seven written questions and three technical quizzes. You may skip a question. Skipped questions receive 0% accuracy. Results support learning and require human review.</div>
<p><label><input id="consent" type="checkbox"> I understand and consent to this prototype assessment.</label></p>
<button onclick="begin()">Begin assessment in full screen</button></section>
<section id="test" class="card hidden"><span id="setPill" class="pill"></span><div class="progress"><i id="bar"></i></div><p id="counter" class="muted"></p><h2 id="question"></h2><pre id="code" class="hidden"></pre><div id="response"></div>
<div class="actions"><button onclick="saveAndContinue()">Save and continue</button><button class="skip" onclick="skipQuestion()">Skip question</button></div><p id="fullscreenWarning" class="warn hidden">Full screen was exited. Select the button below to return.</p><button id="returnFull" class="secondary hidden" onclick="enterFullscreen()">Return to full screen</button></section>
</main><script>
const token=__TOKEN__;let assessment=null,index=0,answers=[];
function show(id){['ready','test'].forEach(x=>document.getElementById(x).classList.toggle('hidden',x!==id))}
async function enterFullscreen(){try{if(!document.fullscreenElement)await document.documentElement.requestFullscreen()}catch(e){}}
async function begin(){if(!document.getElementById('consent').checked){alert('Please confirm consent.');return}await enterFullscreen();assessment=await api('/api/start',{token});show('test');document.getElementById('setPill').textContent=assessment.role_name+' - Set '+assessment.set_id;render()}
function render(){const q=assessment.questions[index];document.getElementById('counter').textContent=`Question ${index+1} of ${assessment.questions.length} - ${q.question_type==='quiz'?'Technical quiz':'Written response'}`;document.getElementById('question').textContent=q.question;document.getElementById('bar').style.width=`${index/assessment.questions.length*100}%`;const code=document.getElementById('code');code.textContent=q.code||'';code.classList.toggle('hidden',!q.code);const box=document.getElementById('response');if(q.question_type==='quiz'){box.innerHTML=q.options.map((o,i)=>`<label class="option"><input type="radio" name="quiz" value="${i}">${esc(o)}</label>`).join('')}else{box.innerHTML='<textarea id="answer" placeholder="Explain the technical concepts in your own words. Grammar and spelling are not graded."></textarea>'}}
function currentResponse(skipped=false){const q=assessment.questions[index];if(skipped)return{question_id:q.question_id,skipped:true};if(q.question_type==='quiz'){const chosen=document.querySelector('input[name="quiz"]:checked');if(!chosen)return null;return{question_id:q.question_id,selected_index:Number(chosen.value),skipped:false}}const value=document.getElementById('answer').value.trim();if(!value)return null;return{question_id:q.question_id,answer:value,skipped:false}}
function saveAndContinue(){const response=currentResponse(false);if(!response){alert('Please answer the question or select Skip question.');return}advance(response)}
function skipQuestion(){if(confirm('Skip this question? It will receive 0% accuracy.'))advance(currentResponse(true))}
async function advance(response){answers.push(response);index++;if(index<assessment.questions.length){render();return}const resultTab=window.open('about:blank','_blank');if(resultTab)resultTab.document.write('<p style="font-family:sans-serif;padding:30px">Evaluating answers and preparing your learning report...</p>');let result;try{result=await api('/api/submit',{token,answers})}catch(e){if(resultTab)resultTab.close();alert(e.message);return}if(document.fullscreenElement)await document.exitFullscreen();const url='/result?id='+encodeURIComponent(result.result_id);if(resultTab)resultTab.location=url;else window.location=url}
document.addEventListener('fullscreenchange',()=>{if(!assessment)return;const exited=!document.fullscreenElement;document.getElementById('fullscreenWarning').classList.toggle('hidden',!exited);document.getElementById('returnFull').classList.toggle('hidden',!exited)});
async function api(url,data){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});const body=await r.json();if(!r.ok)throw new Error(body.error||'Request failed');return body}
function esc(s){return String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
</script></body></html>'''

RESULT_HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SkillScope Results</title><style>__STYLE__</style></head><body><header><b>SkillScope AI</b><span>Development report</span></header><main><section class="card"><span class="pill">Human review required</span><h1>Your assessment result</h1><div id="content"><p>Loading result...</p></div></section></main><script>
const id=new URLSearchParams(location.search).get('id');fetch('/api/result?id='+encodeURIComponent(id)).then(async r=>{const b=await r.json();if(!r.ok)throw new Error(b.error||'Unable to load result');return b}).then(render).catch(e=>document.getElementById('content').textContent=e.message);
function render(r){let html=`<div class="score">${r.average_accuracy}%</div><p><b>Overall result:</b> ${esc(r.overall_level)}</p><p>${r.competent_count} of ${r.question_count} questions reached the 80% competency threshold. ${r.skipped_count} question(s) skipped.</p><p class="muted">Accuracy is based on approved technical concepts or the correct quiz option. Grammar is not graded. AI feedback is developmental and requires technical-director review.</p><h2>Question-by-question feedback</h2>`;for(const x of r.questions){const status=x.skipped?'Skipped':(x.competent?'Competency demonstrated':'Development needed');html+=`<article class="result-item"><span class="pill">${esc(x.question_type)}</span><h3>${esc(x.question)}</h3>${x.code?`<pre>${esc(x.code)}</pre>`:''}<p><b>Your answer:</b> ${esc(x.employee_answer)}</p><p class="${x.competent?'good':'bad'}"><b>Accuracy:</b> ${x.accuracy}% - ${status}</p><p><b>Expected answer or reference:</b> ${esc(x.expected_answer)}</p>${x.missing_concepts.length?`<p><b>Concepts to improve:</b> ${x.missing_concepts.map(esc).join(', ')}</p>`:''}${x.rag_feedback?`<div class="evidence"><b>RAG learning feedback:</b> ${esc(x.rag_feedback)}<br><span class="muted">Mode: ${esc(x.rag_mode)}</span></div>`:''}<p><a href="${esc(x.learning_url)}" target="_blank" rel="noopener">Study this approved reference</a></p></article>`}document.getElementById('content').innerHTML=html}
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
</script></body></html>'''

HOME_HTML = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>SkillScope AI</title><style>{BASE_STYLE}</style></head><body><header><b>SkillScope AI</b></header><main><section class='card'><h1>Invitation link required</h1><p>Candidates enter through a signed role-specific link. Generate one from the project folder:</p><pre>python create_invitation.py --participant EMP-001 --role qa_test_engineer</pre><p class='muted'>The candidate will not select a role or question set.</p></section></main></body></html>"""


def public_question(row: dict) -> dict:
    output = {"question_id": row["question_id"], "question": row["question"], "question_type": row.get("question_type", "written")}
    if output["question_type"] == "quiz":
        output.update({"code": row.get("code", ""), "options": row["options"]})
    return output


def expected_answer(question: dict) -> str:
    if question.get("question_type") == "quiz":
        return f"{question['options'][question['correct_index']]}. {question['explanation']}"
    return CHUNK_BY_ID.get(question.get("reference_chunk_id"), {}).get("text", "Review the approved reference and required concepts.")


def developmental_feedback(question: dict, answer: str) -> tuple[str, str]:
    result = RAG.evaluate_with_safe_fallback(question, answer)
    areas = result.get("improvement_areas", [])
    if areas:
        return " ".join(filter(None, [areas[0].get("reason", ""), areas[0].get("recommended_activity", "")])), result["mode"]
    missing = result.get("missing_concepts", [])
    return (("Review these concepts: " + ", ".join(missing)) if missing else result.get("rationale", "Human review required.")), result["mode"]


class Handler(BaseHTTPRequestHandler):
    def send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode(); self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def send_html(self, html: str, status: int = 200) -> None:
        body = html.encode(); self.send_response(status); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/": self.send_html(HOME_HTML); return
        if parsed.path == "/assessment":
            token = parse_qs(parsed.query).get("token", [""])[0]
            try:
                invite = verify_invitation_token(token); role_name = ROLES[invite["role"]]
            except (ValueError, KeyError):
                self.send_html("<h1>Invalid or expired assessment link</h1>", 400); return
            html = CANDIDATE_HTML.replace("__STYLE__", BASE_STYLE).replace("__TOKEN__", json.dumps(token)).replace("__PARTICIPANT__", escape(str(invite["participant_id"]))).replace("__ROLE__", escape(role_name))
            self.send_html(html); return
        if parsed.path == "/result": self.send_html(RESULT_HTML.replace("__STYLE__", BASE_STYLE)); return
        if parsed.path == "/api/result":
            result_id = parse_qs(parsed.query).get("id", [""])[0]
            self.send_json(RESULTS[result_id]) if result_id in RESULTS else self.send_json({"error": "Result not found or server restarted."}, 404); return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        try:
            size = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(size) or b"{}")
            invite = verify_invitation_token(payload["token"]); role = invite["role"]
            assigned = assign_set(invite["participant_id"], role, SETS)
            if self.path == "/api/start":
                self.send_json({"role": role, "role_name": ROLES[role], "set_id": assigned["set_id"], "questions": [public_question(q) for q in assigned["questions"]]}); return
            if self.path == "/api/submit":
                assigned_by_id = {q["question_id"]: q for q in assigned["questions"]}
                supplied = {row["question_id"]: row for row in payload.get("answers", []) if row.get("question_id") in assigned_by_id}
                details = []
                for question in assigned["questions"]:
                    response = supplied.get(question["question_id"], {"question_id": question["question_id"], "skipped": True}); scored = score_response(response, question)
                    if question.get("question_type") == "quiz":
                        chosen = response.get("selected_index"); employee_answer = "Skipped" if scored["skipped"] else question["options"][int(chosen)]
                    else: employee_answer = "Skipped" if scored["skipped"] else str(response.get("answer", ""))
                    details.append({"question_id": question["question_id"], "question": question["question"], "question_type": question.get("question_type", "written"), "code": question.get("code", ""), "employee_answer": employee_answer, "expected_answer": expected_answer(question), "learning_url": question["reference_source_url"], "accuracy": scored["accuracy"], "competent": scored["competent"], "skipped": scored["skipped"], "missing_concepts": scored["missing_concepts"], "rag_feedback": "", "rag_mode": "not_requested"})
                weak = sorted([x for x in details if x["question_type"] == "written" and not x["skipped"] and not x["competent"]], key=lambda x: x["accuracy"])[:RAG_LIMIT]
                for item in weak:
                    item["rag_feedback"], item["rag_mode"] = developmental_feedback(assigned_by_id[item["question_id"]], item["employee_answer"])
                average = round(sum(x["accuracy"] for x in details) / len(details), 1) if details else 0.0; result_id = uuid.uuid4().hex
                RESULTS[result_id] = {"participant_id": invite["participant_id"], "role": ROLES[role], "set_id": assigned["set_id"], "average_accuracy": average, "overall_level": "Knowledge demonstrated" if average >= 80 else "Development recommended" if average >= 40 else "Priority learning recommended", "competent_count": sum(x["competent"] for x in details), "skipped_count": sum(x["skipped"] for x in details), "question_count": len(details), "human_review_required": True, "questions": details}
                self.send_json({"result_id": result_id}); return
            self.send_error(404)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError, IndexError) as exc:
            self.send_json({"error": str(exc)}, 400)

    def log_message(self, format: str, *args) -> None: return


if __name__ == "__main__":
    print("SkillScope AI v3: http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
