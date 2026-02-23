from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from src.api.main import _get_llm, _set_llm
from src.interview.agent import _extract_and_update, process_turn
from src.interview.state import InterviewState


def _real_llm():
    if not (os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Real model key is required for integration tests.")
    _set_llm(None)
    return _get_llm()


def test_extract_and_update_with_real_model():
    llm = _real_llm()
    state = InterviewState()
    history = [HumanMessage(content="我叫张三，做Python开发3年")]
    updated = _extract_and_update(history, state, llm)
    assert isinstance(updated, InterviewState)


def test_process_turn_with_real_model():
    llm = _real_llm()
    state = InterviewState()
    history = [HumanMessage(content="你好，我叫王泽")]
    reply, new_state = process_turn(history, state, llm)
    assert isinstance(reply, str) and reply
    assert isinstance(new_state, InterviewState)
