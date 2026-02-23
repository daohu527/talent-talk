import os
from fastapi.testclient import TestClient


def _require_real_key():
    if not (os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Real model key is required for integration tests.")


def test_demo_api_returns_structured_fields_with_real_model():
    _require_real_key()
    from src.api.main import app, _set_llm

    _set_llm(None)
    client = TestClient(app)

    resp = client.post("/demo_api", json={"text": "我是王泽，做算法的。"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("next_question"), str) and data["next_question"]
    assert "session_id" in data and data["session_id"]
    assert "phase" in data


def test_demo_api_multi_turn_with_real_model():
    _require_real_key()
    from src.api.main import app, _set_llm

    _set_llm(None)
    client = TestClient(app)

    r1 = client.post("/demo_api", json={"text": "我是王泽，做算法的。"})
    assert r1.status_code == 200, r1.text
    d1 = r1.json()
    sid = d1.get("session_id")
    assert sid

    r2 = client.post("/demo_api", json={"text": "我擅长 Python 和 PyTorch。", "session_id": sid})
    assert r2.status_code == 200, r2.text
    d2 = r2.json()
    assert d2.get("session_id") == sid
    assert isinstance(d2.get("next_question"), str) and d2["next_question"]


def test_demo_api_bootstrap_first_question_with_real_model():
    _require_real_key()
    from src.api.main import app, _set_llm

    _set_llm(None)
    client = TestClient(app)

    resp = client.post("/demo_api", json={"text": ""})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert isinstance(data.get("next_question"), str) and data["next_question"]
    assert data.get("session_id")
