from __future__ import annotations

import os

from fastapi.testclient import TestClient

from src.api.main import app, _set_llm

client = TestClient(app)


def _require_real_key():
    if not (os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Real model key is required for integration tests.")


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_reply_real_model():
    _require_real_key()
    _set_llm(None)
    payload = {
        "history": [],
        "state": {},
        "user_message": "你好",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("reply"), str)
    assert isinstance(data.get("state"), dict)
    assert isinstance(data.get("finished"), bool)


def test_chat_invalid_state_returns_422():
    payload = {
        "history": [],
        "state": {"finished": "not-a-bool"},
        "user_message": "hi",
    }
    resp = client.post("/chat", json=payload)
    assert resp.status_code == 422


def test_report_returns_markdown_real_model():
    _require_real_key()
    _set_llm(None)
    payload = {
        "history": [{"role": "human", "content": "你好，我叫张三，做Python开发3年"}],
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
    assert resp.status_code == 200, resp.text
    md = resp.json().get("markdown", "")
    assert isinstance(md, str) and md
    assert "#" in md


def test_report_rejects_unfinished_interview():
    payload = {
        "history": [],
        "state": {"finished": False},
    }
    resp = client.post("/report", json=payload)
    assert resp.status_code == 400
