
import os
import json
from typing import TypedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# 1. GEMINI CONFIGURATION
# ============================================================

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not configured.")


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2
)


# ============================================================
# 2. LANGGRAPH STATE
# ============================================================

class CrewState(TypedDict, total=False):
    task: str
    code: str
    tests: dict
    manager_report: dict
    status: str


# ============================================================
# 3. HELPERS
# ============================================================

def extract_text(response):

    content = getattr(response, "content", "")

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def clean_code(text):

    text = text.strip()

    if "```python" in text:
        text = text.replace("```python", "", 1)

    if "```" in text:
        text = text.replace("```", "")

    return text.strip()


def parse_json_response(text):

    text = text.strip()

    if "```json" in text:
        text = text.replace("```json", "", 1)

    if "```" in text:
        text = text.replace("```", "")

    text = text.strip()

    try:
        return json.loads(text)

    except Exception:

        return {
            "summary": text,
            "items": [],
            "status": "Review completed"
        }


# ============================================================
# 4. DEVELOPER AGENT
# ============================================================

def developer_node(state: CrewState):

    task = state["task"]

    prompt = f"""
You are the Developer Agent of DevCrew AI.

User task:
{task}

Create a clean Python solution.

Requirements:
- Understand the task.
- Write practical Python code.
- Use functions where appropriate.
- Use meaningful names.
- Include useful validation.
- Include error handling.
- Return ONLY Python code.
"""

    response = llm.invoke(prompt)

    code = clean_code(
        extract_text(response)
    )

    return {
        "code": code,
        "status": "Developer completed"
    }


# ============================================================
# 5. TESTER AGENT
# ============================================================

def tester_node(state: CrewState):

    task = state["task"]
    code = state["code"]

    prompt = f"""
You are the Tester / QA Agent of DevCrew AI.

Original task:
{task}

Developer code:
{code}

Perform static code review.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "overall_status": "PASS / PASS WITH WARNINGS / NEEDS FIXES",
    "summary": "short summary",
    "code_quality": "short assessment",
    "functionality": "short assessment",
    "security": "short assessment",
    "issues": [
        "issue 1",
        "issue 2"
    ],
    "test_scenarios": [
        "scenario 1",
        "scenario 2",
        "scenario 3"
    ],
    "recommendations": [
        "recommendation 1",
        "recommendation 2"
    ]
}}

Important:
- Do NOT execute arbitrary code.
- Perform static analysis only.
- Keep each item concise.
"""

    response = llm.invoke(prompt)

    report = parse_json_response(
        extract_text(response)
    )

    return {
        "tests": report,
        "status": "Tester completed"
    }


# ============================================================
# 6. MANAGER AGENT
# ============================================================

def manager_node(state: CrewState):

    task = state["task"]
    code = state["code"]
    tests = state["tests"]

    prompt = f"""
You are the Engineering Manager Agent of DevCrew AI.

Original task:
{task}

Developer code:
{code}

Tester report:
{json.dumps(tests, indent=2)}

Review the complete work.

Return ONLY valid JSON.

Use exactly this structure:

{{
    "overall_status": "READY / READY WITH MINOR FIXES / NEEDS REVISION",
    "task_understanding": "short assessment",
    "development_summary": "short assessment",
    "qa_summary": "short assessment",
    "major_issues": [
        "issue 1",
        "issue 2"
    ],
    "recommendations": [
        "recommendation 1",
        "recommendation 2"
    ],
    "production_readiness": "short assessment",
    "final_note": "short final decision"
}}

Keep every field concise and professional.
"""

    response = llm.invoke(prompt)

    report = parse_json_response(
        extract_text(response)
    )

    return {
        "manager_report": report,
        "status": "Manager completed"
    }


# ============================================================
# 7. ARCHIVER AGENT
# ============================================================

def archiver_node(state: CrewState):

    return {
        "status": "DevCrew workflow completed successfully"
    }


# ============================================================
# 8. LANGGRAPH WORKFLOW
# ============================================================

workflow = StateGraph(CrewState)

workflow.add_node(
    "developer",
    developer_node
)

workflow.add_node(
    "tester",
    tester_node
)

workflow.add_node(
    "manager",
    manager_node
)

workflow.add_node(
    "archiver",
    archiver_node
)

workflow.add_edge(
    START,
    "developer"
)

workflow.add_edge(
    "developer",
    "tester"
)

workflow.add_edge(
    "tester",
    "manager"
)

workflow.add_edge(
    "manager",
    "archiver"
)

workflow.add_edge(
    "archiver",
    END
)

crew = workflow.compile()


# ============================================================
# 9. FASTAPI
# ============================================================

app = FastAPI(
    title="DevCrew AI",
    description="AI-powered multi-agent development crew",
    version="1.0.0"
)


class TaskRequest(BaseModel):

    task: str


# ============================================================
# 10. FRONTEND
# ============================================================

HTML_PAGE = """

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>DevCrew AI</title>

<style>

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {

    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    color: #f8fafc;

    min-height: 100vh;

    background:
        radial-gradient(
            circle at 15% 10%,
            #172554,
            transparent 35%
        ),
        radial-gradient(
            circle at 90% 20%,
            #0c4a6e,
            transparent 30%
        ),
        #020617;
}

.container {

    width: min(1200px, 92%);

    margin: auto;

    padding: 35px 0 70px;
}


/* HEADER */

.header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 70px;
}

.logo {

    font-size: 27px;

    font-weight: 800;

    letter-spacing: -1px;
}

.logo span {

    color: #38bdf8;
}

.status {

    display: flex;

    align-items: center;

    gap: 8px;

    padding: 8px 14px;

    border-radius: 999px;

    background:
        rgba(34,197,94,0.08);

    border:
        1px solid rgba(34,197,94,0.2);

    color: #bbf7d0;

    font-size: 12px;
}

.status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #22c55e;

    box-shadow:
        0 0 12px #22c55e;
}


/* HERO */

.hero {

    text-align: center;

    margin-bottom: 45px;
}

.hero h1 {

    font-size:
        clamp(42px, 7vw, 76px);

    letter-spacing: -4px;

    margin-bottom: 18px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #7dd3fc
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.hero p {

    max-width: 680px;

    margin: auto;

    color: #94a3b8;

    font-size: 17px;

    line-height: 1.7;
}


/* INPUT */

.card {

    background:
        rgba(15,23,42,0.75);

    border:
        1px solid rgba(148,163,184,0.12);

    border-radius: 22px;

    padding: 28px;

    margin-bottom: 24px;

    backdrop-filter: blur(18px);

    box-shadow:
        0 20px 70px
        rgba(0,0,0,0.25);
}

.card-title {

    font-size: 18px;

    font-weight: 700;

    margin-bottom: 16px;
}

textarea {

    width: 100%;

    min-height: 150px;

    resize: vertical;

    background: #020617;

    color: white;

    border:
        1px solid #1e293b;

    border-radius: 14px;

    padding: 18px;

    font-size: 15px;

    line-height: 1.6;

    outline: none;
}

textarea:focus {

    border-color: #38bdf8;

    box-shadow:
        0 0 0 3px
        rgba(56,189,248,0.08);
}

button {

    width: 100%;

    margin-top: 18px;

    padding: 15px;

    border: none;

    border-radius: 14px;

    color: white;

    font-size: 16px;

    font-weight: 700;

    cursor: pointer;

    background:
        linear-gradient(
            90deg,
            #0284c7,
            #2563eb
        );
}

button:hover {

    box-shadow:
        0 12px 30px
        rgba(37,99,235,0.3);
}

button:disabled {

    opacity: 0.6;

    cursor: wait;
}


/* WORKFLOW */

.workflow {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 14px;

    margin-bottom: 25px;
}

.agent {

    text-align: center;

    padding: 20px;

    border-radius: 18px;

    background:
        rgba(15,23,42,0.7);

    border:
        1px solid #1e293b;
}

.agent-number {

    width: 40px;

    height: 40px;

    margin:
        0 auto 12px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 50%;

    background: #0f172a;

    color: #38bdf8;

    font-weight: 800;
}

.agent h3 {

    margin-bottom: 6px;

    font-size: 15px;
}

.agent p {

    color: #64748b;

    font-size: 12px;
}


/* RESULTS */

.results {

    display: none;
}

.result-header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    margin-bottom: 20px;
}

.success-badge {

    padding: 7px 12px;

    border-radius: 999px;

    background:
        rgba(34,197,94,0.1);

    color: #86efac;

    font-size: 12px;

    border:
        1px solid
        rgba(34,197,94,0.2);
}


/* AGENT STATUS */

.agent-status-grid {

    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 12px;

    margin-bottom: 24px;
}

.status-card {

    padding: 16px;

    border-radius: 15px;

    background: #020617;

    border:
        1px solid #1e293b;
}

.status-card .label {

    color: #64748b;

    font-size: 11px;

    margin-bottom: 6px;
}

.status-card .value {

    font-size: 13px;

    font-weight: 700;

    color: #86efac;
}


/* CODE */

.code-box {

    background: #020617;

    border:
        1px solid #1e293b;

    border-radius: 15px;

    padding: 22px;

    overflow-x: auto;
}

pre {

    white-space: pre-wrap;

    color: #cbd5e1;

    font-family:
        "Courier New",
        monospace;

    font-size: 13px;

    line-height: 1.7;
}


/* REPORT GRID */

.info-grid {

    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 15px;

    margin-top: 15px;
}

.info-card {

    background: #020617;

    border:
        1px solid #1e293b;

    border-radius: 15px;

    padding: 20px;
}

.info-card h4 {

    font-size: 14px;

    margin-bottom: 10px;

    color: #7dd3fc;
}

.info-card p {

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.6;
}


/* LISTS */

.list {

    display: flex;

    flex-direction: column;

    gap: 9px;
}

.list-item {

    padding: 11px 13px;

    border-radius: 10px;

    background:
        rgba(30,41,59,0.45);

    color: #cbd5e1;

    font-size: 13px;

    line-height: 1.5;
}

.list-item::before {

    content: "• ";

    color: #38bdf8;

    font-weight: bold;
}


/* RESPONSIVE */

@media (max-width: 800px) {

    .workflow,
    .agent-status-grid {

        grid-template-columns:
            repeat(2, 1fr);
    }

    .info-grid {

        grid-template-columns: 1fr;
    }
}

@media (max-width: 500px) {

    .workflow,
    .agent-status-grid {

        grid-template-columns: 1fr;
    }
}

</style>

</head>


<body>


<div class="container">


<header class="header">

<div class="logo">

DEV<span>CREW</span> AI

</div>


<div class="status">

<span class="status-dot"></span>

SYSTEM ONLINE

</div>

</header>


<section class="hero">

<h1>

Build with an AI Crew.

</h1>

<p>

DevCrew AI orchestrates specialized AI agents
to develop, test, review, and finalize software tasks.

</p>

</section>


<section class="card">

<div class="card-title">

Describe your development task

</div>


<textarea
id="task"
placeholder="Example: Build a Python program that analyzes student marks and generates a performance report..."
></textarea>


<button
id="runBtn"
onclick="runCrew()"
>

Run DevCrew →

</button>

</section>


<section class="workflow">


<div class="agent">

<div class="agent-number">01</div>

<h3>Developer</h3>

<p>Builds the solution</p>

</div>


<div class="agent">

<div class="agent-number">02</div>

<h3>Tester</h3>

<p>Reviews quality & risks</p>

</div>


<div class="agent">

<div class="agent-number">03</div>

<h3>Manager</h3>

<p>Reviews the outcome</p>

</div>


<div class="agent">

<div class="agent-number">04</div>

<h3>Archiver</h3>

<p>Finalizes the workflow</p>

</div>


</section>


<section
id="results"
class="results"
>


<div class="card">


<div class="result-header">

<div class="card-title">

DevCrew Results

</div>

<div class="success-badge">

✓ Workflow Completed

</div>

</div>


<div class="agent-status-grid">


<div class="status-card">

<div class="label">AGENT 01</div>

<div class="value">✓ Developer</div>

</div>


<div class="status-card">

<div class="label">AGENT 02</div>

<div class="value">✓ Tester</div>

</div>


<div class="status-card">

<div class="label">AGENT 03</div>

<div class="value">✓ Manager</div>

</div>


<div class="status-card">

<div class="label">AGENT 04</div>

<div class="value">✓ Archiver</div>

</div>


</div>


</div>


<div class="card">


<div class="card-title">

💻 Generated Code

</div>


<div class="code-box">

<pre id="code"></pre>

</div>


</div>


<div class="card">


<div class="card-title">

🧪 QA Analysis

</div>


<div class="info-grid">


<div class="info-card">

<h4>Overall Status</h4>

<p id="qaStatus"></p>

</div>


<div class="info-card">

<h4>Summary</h4>

<p id="qaSummary"></p>

</div>


<div class="info-card">

<h4>Code Quality</h4>

<p id="codeQuality"></p>

</div>


<div class="info-card">

<h4>Functionality</h4>

<p id="functionality"></p>

</div>


<div class="info-card">

<h4>Security</h4>

<p id="security"></p>

</div>


</div>


<div class="info-card"
style="margin-top:15px;">

<h4>⚠️ Issues Found</h4>

<div
id="issues"
class="list"
></div>

</div>


<div class="info-card"
style="margin-top:15px;">

<h4>💡 Recommendations</h4>

<div
id="recommendations"
class="list"
></div>

</div>


<div class="info-card"
style="margin-top:15px;">

<h4>🧪 Test Scenarios</h4>

<div
id="scenarios"
class="list"
></div>

</div>


</div>


<div class="card">


<div class="card-title">

👨‍💼 Manager Review

</div>


<div class="info-grid">


<div class="info-card">

<h4>Overall Status</h4>

<p id="managerStatus"></p>

</div>


<div class="info-card">

<h4>Task Understanding</h4>

<p id="taskUnderstanding"></p>

</div>


<div class="info-card">

<h4>Development Summary</h4>

<p id="developmentSummary"></p>

</div>


<div class="info-card">

<h4>QA Summary</h4>

<p id="qaSummaryManager"></p>

</div>


<div class="info-card">

<h4>Production Readiness</h4>

<p id="productionReadiness"></p>

</div>


</div>


<div class="info-card"
style="margin-top:15px;">

<h4>⚠️ Major Issues</h4>

<div
id="majorIssues"
class="list"
></div>

</div>


<div class="info-card"
style="margin-top:15px;">

<h4>💡 Recommendations</h4>

<div
id="managerRecommendations"
class="list"
></div>

</div>


<div class="info-card"
style="margin-top:15px;">

<h4>Final Note</h4>

<p id="finalNote"></p>

</div>


</div>


</section>


</div>


<script>


function setText(id, value) {

    document.getElementById(id).textContent =
        value || "Not provided";

}


function setList(id, items) {

    const element =
        document.getElementById(id);

    element.innerHTML = "";

    if (!Array.isArray(items) || items.length === 0) {

        const item =
            document.createElement("div");

        item.className = "list-item";

        item.textContent = "No issues identified.";

        element.appendChild(item);

        return;
    }


    items.forEach(function(itemText) {

        const item =
            document.createElement("div");

        item.className = "list-item";

        item.textContent = itemText;

        element.appendChild(item);

    });

}


async function runCrew() {


    const task =
        document
        .getElementById("task")
        .value
        .trim();


    const button =
        document
        .getElementById("runBtn");


    const results =
        document
        .getElementById("results");


    if (!task) {

        alert(
            "Please describe a development task."
        );

        return;
    }


    button.disabled = true;

    button.innerText =
        "DevCrew is working...";


    results.style.display =
        "none";


    try {


        const response =
            await fetch(
                "/api/run",
                {

                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        task: task
                    })

                }
            );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Something went wrong."
            );

        }


        const qa =
            data.tests || {};


        const manager =
            data.manager_report || {};


        setText(
            "code",
            data.code
        );


        setText(
            "qaStatus",
            qa.overall_status
        );


        setText(
            "qaSummary",
            qa.summary
        );


        setText(
            "codeQuality",
            qa.code_quality
        );


        setText(
            "functionality",
            qa.functionality
        );


        setText(
            "security",
            qa.security
        );


        setList(
            "issues",
            qa.issues
        );


        setList(
            "recommendations",
            qa.recommendations
        );


        setList(
            "scenarios",
            qa.test_scenarios
        );


        setText(
            "managerStatus",
            manager.overall_status
        );


        setText(
            "taskUnderstanding",
            manager.task_understanding
        );


        setText(
            "developmentSummary",
            manager.development_summary
        );


        setText(
            "qaSummaryManager",
            manager.qa_summary
        );


        setText(
            "productionReadiness",
            manager.production_readiness
        );


        setList(
            "majorIssues",
            manager.major_issues
        );


        setList(
            "managerRecommendations",
            manager.recommendations
        );


        setText(
            "finalNote",
            manager.final_note
        );


        results.style.display =
            "block";


        results.scrollIntoView({
            behavior: "smooth"
        });


    }

    catch (error) {

        alert(error.message);

    }

    finally {

        button.disabled = false;

        button.innerText =
            "Run DevCrew →";

    }

}


</script>


</body>

</html>

"""


# ============================================================
# 11. API ROUTES
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML_PAGE


@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "DevCrew AI"
    }


@app.post("/api/run")
def run_crew(request: TaskRequest):

    if not request.task.strip():

        raise HTTPException(
            status_code=400,
            detail="Task cannot be empty."
        )


    initial_state: CrewState = {

        "task": request.task,

        "code": "",

        "tests": {},

        "manager_report": {},

        "status": "Starting DevCrew"

    }


    result = crew.invoke(
        initial_state
    )


    return {

        "code":
            result.get(
                "code",
                ""
            ),

        "tests":
            result.get(
                "tests",
                {}
            ),

        "manager_report":
            result.get(
                "manager_report",
                {}
            ),

        "status":
            result.get(
                "status",
                ""
            )

    }
