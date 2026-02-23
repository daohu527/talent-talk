from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from src.api.main import _get_llm, _set_llm
from src.orchestration.langgraph import GraphRunner
from src.interview.state import InterviewState


def _real_llm():
    if not (os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Real model key is required for integration tests.")
    _set_llm(None)
    return _get_llm()


def test_graph_runner_with_real_model():
    llm = _real_llm()
    runner = GraphRunner(llm)
    history = [HumanMessage(content="我叫李四，做后端开发")]
    state = InterviewState()
    out = runner.run_once(history, state)
    assert isinstance(out.get("reply"), str)
    assert "question_for" in out
