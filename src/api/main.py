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
from typing import Any

from fastapi import FastAPI, HTTPException
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from src.interview.agent import process_turn
from src.interview.report import generate_report
from src.interview.state import InterviewState

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


# ---------------------------------------------------------------------------
# LLM singleton (lazily initialised)
# ---------------------------------------------------------------------------

_llm: Any = None


def _get_llm() -> Any:
    global _llm
    if _llm is None:
        from langchain_openai import ChatOpenAI  # imported lazily

        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0.5,
            api_key=os.getenv("OPENAI_API_KEY", ""),
        )
    return _llm


def _set_llm(llm: Any) -> None:
    """Allow tests to inject a mock LLM."""
    global _llm
    _llm = llm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_lc_messages(messages: list[Message]) -> list:
    """Convert API message dtos to LangChain message objects."""
    result = []
    for m in messages:
        if m.role == "human":
            result.append(HumanMessage(content=m.content))
        elif m.role == "ai":
            result.append(AIMessage(content=m.content))
        elif m.role == "system":
            result.append(SystemMessage(content=m.content))
    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
