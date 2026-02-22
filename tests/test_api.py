"""Tests for the FastAPI application endpoints."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import src.api.main as main_module
from src.api.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Mock LLM helpers
# ---------------------------------------------------------------------------


class DualMockLLM:
    """Returns different responses for successive invocations."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, messages):
        content = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return AIMessage(content=content)


def _inject(llm):
    main_module._set_llm(llm)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /chat
# ---------------------------------------------------------------------------


def test_chat_returns_reply():
    _inject(DualMockLLM(["{}", "请问您叫什么名字？"]))
    payload = {
        "history": [],
        "state": {},
        "user_message": "你好",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert isinstance(data["reply"], str)
    assert "state" in data
    assert "finished" in data


def test_chat_marks_finished_on_end_marker():
    _inject(DualMockLLM(["{}", "面试结束！[END]"]))
    payload = {
        "history": [],
        "state": {
            "name": "张三",
            "experience_years": "3年",
            "tech_stack": "Python",
            "biggest_project": "电商平台",
            "expected_salary": "20k",
        },
        "user_message": "好的",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200
    assert resp.json()["finished"] is True


def test_chat_invalid_state_returns_422():
    _inject(DualMockLLM(["{}", "hello"]))
    payload = {
        "history": [],
        "state": {"finished": "not-a-bool"},
        "user_message": "hi",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /report
# ---------------------------------------------------------------------------


_SAMPLE_EXTRACTION = json.dumps({
    "candidate_name": "张三",
    "summary": "优秀候选人",
    "skills": ["Python"],
    "experience_years": "3年",
    "biggest_project": "电商平台",
    "expected_salary": "20k",
    "communication_score": 9,
    "suggested_level": "P6",
})


def test_report_returns_markdown():
    _inject(DualMockLLM([_SAMPLE_EXTRACTION]))
    payload = {
        "history": [{"role": "human", "content": "你好"}],
        "state": {
            "name": "张三",
            "experience_years": "3年",
            "tech_stack": "Python",
            "biggest_project": "电商平台",
            "expected_salary": "20k",
            "finished": True,
        },
    }
    resp = client.post("/report", json=payload)
    assert resp.status_code == 200
    md = resp.json()["markdown"]
    assert "张三" in md
    assert "#" in md


def test_report_rejects_unfinished_interview():
    _inject(DualMockLLM(["{}"]))
    payload = {
        "history": [],
        "state": {"finished": False},
    }
    resp = client.post("/report", json=payload)
    assert resp.status_code == 400
