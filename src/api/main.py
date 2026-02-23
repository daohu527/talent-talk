"""FastAPI application exposing the TalentTalk interview API.

Endpoints
---------
POST /chat          – Send a candidate message and receive the AI reply.
POST /report        – Generate a Markdown report for a finished session.
GET  /health        – Liveness probe.

The LLM is initialised lazily so that the app can start without API keys
set (useful for unit tests where the LLM is monkeypatched).
"""

from __future__ import annotations

import os
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.interview.agent import process_turn
from src.interview.dual_agent import DualAgentInterviewSystem
from src.interview.report import generate_report
from src.interview.state import InterviewState
from src.api.session_store import get_store

app = FastAPI(title="TalentTalk", version="0.1.0")

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class Message(BaseModel):
    role: str  # "human" or "ai"
    content: str


class ChatRequest(BaseModel):
    history: list[Message]
    state: dict[str, Any]
    user_message: str


class ChatResponse(BaseModel):
    reply: str
    state: dict[str, Any]
    finished: bool


class ReportRequest(BaseModel):
    history: list[Message]
    state: dict[str, Any]


class ReportResponse(BaseModel):
    markdown: str


# Simple demo DTOs used by tests
class DemoRequest(BaseModel):
    text: str = ""
    session_id: Optional[str] = None


class DemoResponse(BaseModel):
    next_question: Any = None
    summary: Any = None
    thought: Any = None
    phase: Any = None
    completion_score: dict[str, int] = {"background": 0, "tech": 0, "project": 0}
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# LLM singleton (lazily initialised)
# ---------------------------------------------------------------------------

_llm: Any = None


def _discover_model_id(preferred_model: Optional[str], api_key: str, base_url: Optional[str]) -> str:
    """Try to discover an available model id from provider when preferred is unavailable."""
    if preferred_model:
        return preferred_model
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        data = getattr(models, "data", []) or []
        if data:
            model_id = getattr(data[0], "id", None)
            if model_id:
                return str(model_id)
    except Exception:
        pass
    # final fallback (may still fail if provider doesn't expose this model)
    return "gpt-4o"


def _list_model_ids(api_key: str, base_url: Optional[str]) -> list[str]:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url)
        models = client.models.list()
        data = getattr(models, "data", []) or []
        ids = [str(getattr(item, "id", "")) for item in data if getattr(item, "id", None)]
        return [model_id for model_id in ids if model_id]
    except Exception:
        return []


def _get_llm() -> Any:
    global _llm
    if _llm is None:
        ark_api_key = os.getenv("ARK_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")
        api_key = ark_api_key or openai_api_key
        if not api_key:
            raise RuntimeError(
                "No real model credentials found. Set ARK_API_KEY (preferred) or OPENAI_API_KEY."
            )

        preferred_model = os.getenv("ARK_MODEL") or os.getenv("OPENAI_MODEL")
        base_url = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3") if ark_api_key else os.getenv("OPENAI_BASE_URL")
        model_name = _discover_model_id(preferred_model, api_key, base_url)

        try:
            from langchain_openai import ChatOpenAI  # imported lazily

            kwargs = {
                "model": model_name,
                "temperature": 0.5,
                "api_key": api_key,
            }
            if base_url:
                kwargs["base_url"] = base_url
            _llm = ChatOpenAI(**kwargs)
        except Exception:
            from openai import OpenAI

            client = OpenAI(api_key=api_key, base_url=base_url)

            class OpenAICompatLLM:
                def __init__(self, sdk_client, model: str):
                    self.sdk_client = sdk_client
                    self.model = model

                def invoke(self, messages):
                    payload_messages = []
                    for msg in messages or []:
                        if isinstance(msg, dict):
                            role = msg.get("role", "user")
                            content = str(msg.get("content", ""))
                        else:
                            role_name = msg.__class__.__name__.lower()
                            if "system" in role_name:
                                role = "system"
                            elif "ai" in role_name:
                                role = "assistant"
                            else:
                                role = "user"
                            content = str(getattr(msg, "content", ""))
                        payload_messages.append({"role": role, "content": content})

                    candidates: list[str] = []
                    if self.model:
                        candidates.append(self.model)
                    env_candidates = os.getenv("ARK_MODEL_CANDIDATES", "")
                    if env_candidates:
                        candidates.extend([item.strip() for item in env_candidates.split(",") if item.strip()])
                    candidates.extend(_list_model_ids(api_key, base_url))

                    seen: set[str] = set()
                    unique_candidates: list[str] = []
                    for model_id in candidates:
                        if model_id not in seen:
                            seen.add(model_id)
                            unique_candidates.append(model_id)

                    last_error: Optional[Exception] = None
                    resp = None
                    for model_id in unique_candidates:
                        try:
                            resp = self.sdk_client.chat.completions.create(
                                model=model_id,
                                messages=payload_messages,
                                temperature=0.5,
                            )
                            self.model = model_id
                            break
                        except Exception as exc:
                            last_error = exc
                            text = str(exc)
                            if "InvalidEndpointOrModel.NotFound" in text or "does not exist" in text:
                                continue
                            raise

                    if resp is None:
                        raise RuntimeError(
                            "No available chat model for current key/base_url. "
                            f"Tried: {unique_candidates}. Last error: {last_error}"
                        )
                    content = (resp.choices[0].message.content or "") if resp.choices else ""
                    return AIMessage(content=content)

            _llm = OpenAICompatLLM(client, model_name)
    return _llm


def _set_llm(llm: Any) -> None:
    """Allow tests to inject a mock LLM."""
    global _llm
    _llm = llm


class _DualAgentLLMAdapter:
    """Adapter to let DualAgent use the existing llm.invoke interface safely."""

    def __init__(self, llm: Any):
        self.llm = llm

    def invoke(self, messages):
        # Prefer calling the LLM with the original messages (preserving roles).
        # Fall back to joining message contents into a single HumanMessage only
        # if the direct call fails (some LLM wrappers expect a single message).
        try:
            return self.llm.invoke(messages)
        except Exception:
            prompt = "\n\n".join([str(m.get("content", "")) for m in messages if isinstance(m, dict)])
            return self.llm.invoke([HumanMessage(content=prompt)])


def _persist_demo_log(session_id: str, payload: dict[str, Any]) -> None:
    log_dir = Path(os.getenv("INTERVIEW_LOG_DIR", "logs/interviews"))
    log_dir.mkdir(parents=True, exist_ok=True)
    file_path = log_dir / f"{session_id}.jsonl"
    record = {"ts": datetime.utcnow().isoformat() + "Z", **payload}
    # Write pretty-printed JSON to the log so entries are human-readable
    # Keep a blank line between entries to visually separate them.
    with file_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_lc_messages(messages: list[Message]) -> list:
    """Convert API message dtos (Message or dict) to LangChain message objects."""
    result = []
    for m in messages:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role == "human":
            result.append(HumanMessage(content=content))
        elif role == "ai":
            result.append(AIMessage(content=content))
        elif role == "system":
            result.append(SystemMessage(content=content))
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo_page() -> HTMLResponse:
    """Serve the extracted demo HTML template if present."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "demo.html")
    try:
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load demo template: {exc}") from exc
    return HTMLResponse(content=content, media_type="text/html")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    """Process one interview turn.

    The caller maintains the full history and state on their side;
    this endpoint is intentionally stateless.
    """
    try:
        current_state = InterviewState(**req.state)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    history = _to_lc_messages(req.history)
    history.append(HumanMessage(content=req.user_message))

    reply, updated_state = process_turn(history, current_state, _get_llm())

    return ChatResponse(
        reply=reply,
        state=updated_state.model_dump(),
        finished=updated_state.finished,
    )


@app.post("/session")
def create_session() -> dict:
    """Create a new in-memory session and return its id."""
    sid = get_store().create(initial_state={})
    return {"session_id": sid}


@app.get("/session/{sid}")
def get_session(sid: str) -> dict:
    s = get_store().get(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")
    return s


@app.post("/session/{sid}/turn")
def session_turn(sid: str, payload: dict) -> dict:
    """Process a turn for an existing session.

    Expected payload: {"user_message": "..."}
    Returns: {reply, state, finished}
    """
    store = get_store()
    s = store.get(sid)
    if s is None:
        raise HTTPException(status_code=404, detail="session not found")

    user_message = payload.get("user_message")
    if not isinstance(user_message, str):
        raise HTTPException(status_code=422, detail="user_message is required")

    # append human message
    s_history = s.get("history", [])
    s_history.append({"role": "human", "content": user_message})

    try:
        current_state = InterviewState(**(s.get("state") or {}))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    history = _to_lc_messages(s_history)
    history.append(HumanMessage(content=user_message))

    reply, updated_state = process_turn(history, current_state, _get_llm())

    # append AI reply
    s_history.append({"role": "ai", "content": reply})
    s["history"] = s_history
    s["state"] = updated_state.model_dump()
    store.update(sid, s)

    return {"reply": reply, "state": s["state"], "finished": updated_state.finished}


@app.post("/demo_api", response_model=DemoResponse)
def demo_api(req: DemoRequest) -> DemoResponse:
    """Dual-agent demo API with stateful session and structured debug trace.

    Flow: user input -> supervisor directive -> interviewer JSON.
    """
    logger = logging.getLogger("uvicorn.error")
    store = get_store()

    sid = req.session_id
    if not sid:
        sid = f"demo-{uuid.uuid4()}"
        store.update(
            sid,
            {
                "history": [],
                "state": {},
                "demo": {
                    "summary": None,
                    "phase": "开始",
                    "ended": False,
                    "logs": [],
                    "round": 0,
                },
            },
        )

    payload = store.get(sid)
    if payload is None:
        raise HTTPException(status_code=404, detail="demo session not found")

    demo_state = payload.get("demo") or {"summary": None, "phase": "开始", "ended": False, "logs": [], "round": 0}

    try:
        llm = _get_llm()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model init failed: {exc}") from exc
    supervisor = _DualAgentLLMAdapter(llm)
    interviewer = _DualAgentLLMAdapter(llm)
    system = DualAgentInterviewSystem(supervisor, interviewer)
    system.load_state(demo_state)

    user_input = (req.text or "").strip()
    bootstrap = False
    if not user_input and demo_state.get("round", 0) == 0:
        bootstrap = True
        # For bootstrap, ask the agents to generate the starter summary and a single
        # education-background follow-up. We pass a concise instruction so the
        # Supervisor/Interviewer produce the desired JSON output instead of
        # hard-coding strings here.
        user_input = (
            "BOOTSTRAP: 作为面试官，请输出 JSON 格式，仅包含键 'summary' 和 'next_question'。"
            " 'summary' 为一段友好的开场白；'next_question' 为一条针对应聘者教育背景的三段式追问，"
            "要求不少于50字。不要输出其他内容。"
        )

    try:
        turn = system.process_turn(user_input)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Model request failed: {exc}") from exc

    # Note: bootstrap handling is performed by passing a bootstrap instruction as
    # the user_input above so the LLMs generate the starter summary and question.
    # Do not hard-code the text here; keep the trace/audit of model outputs intact.
    history = payload.get("history") or []
    if not bootstrap:
        history.append({"role": "human", "content": req.text})
    # Present a single AI message that includes the short summary (context)
    # followed by the next question. This prevents the frontend from
    # rendering two separate AI messages at the start of a demo session.
    ai_message = (turn.summary or "")
    if ai_message:
        ai_message = ai_message.strip() + "\n\n"
    ai_message += (turn.next_question or "")
    history.append({"role": "ai", "content": ai_message})

    next_state = system.dump_state()
    next_state["round"] = int(demo_state.get("round", 0)) + 1

    payload["history"] = history
    payload["demo"] = next_state
    store.update(sid, payload)

    structured_log = {
        "event": "demo_api_turn",
        "session_id": sid,
        "round": next_state["round"],
        "user_answer": req.text,
        "trace": system.last_trace,
        "output_to_user": {
            "thought": turn.thought,
            "next_question": turn.next_question,
            "summary": turn.summary,
            "phase": turn.phase,
            "completion_score": turn.completion_score,
            "ended": turn.ended,
        },
    }
    # Log a pretty-printed structured trace for easier debugging in logs
    try:
        logger.info(json.dumps(structured_log, ensure_ascii=False, indent=2))
    except Exception:
        logger.info(json.dumps(structured_log, ensure_ascii=False))
    _persist_demo_log(sid, structured_log)

    return DemoResponse(
        next_question=turn.next_question,
        summary=turn.summary,
        thought=turn.thought,
        phase=turn.phase,
        completion_score=turn.completion_score,
        session_id=sid,
    )


@app.get("/demo_session/{sid}/evaluation")
def demo_session_evaluation(sid: str) -> dict:
    session = get_store().get(sid)
    if session is None:
        raise HTTPException(status_code=404, detail="demo session not found")

    demo = session.get("demo") or {}
    score = demo.get("completion_score") or {"background": 0, "tech": 0, "project": 0}
    total = int(score.get("background", 0)) + int(score.get("tech", 0)) + int(score.get("project", 0))
    level = "优秀" if total >= 24 else "良好" if total >= 16 else "待加强"

    return {
        "session_id": sid,
        "phase": demo.get("phase", "开始"),
        "summary": demo.get("summary"),
        "completion_score": score,
        "overall": {
            "total": total,
            "level": level,
            "advice": "可继续补充项目量化结果与技术决策依据" if total < 24 else "可进入薪资与入职时间沟通",
        },
        "logs_file": str(Path(os.getenv("INTERVIEW_LOG_DIR", "logs/interviews")) / f"{sid}.jsonl"),
    }


@app.post("/report", response_model=ReportResponse)
def report(req: ReportRequest) -> ReportResponse:
    """Generate a Markdown interview report from the full session history."""
    try:
        state = InterviewState(**req.state)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if not state.finished:
        raise HTTPException(
            status_code=400,
            detail="Interview is not finished yet. Set state.finished=true first.",
        )

    history = _to_lc_messages(req.history)
    markdown = generate_report(history, _get_llm(), state)
    return ReportResponse(markdown=markdown)
