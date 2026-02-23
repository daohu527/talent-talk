from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from src.api.main import _get_llm, _set_llm
from src.interview.dual_agent import DualAgentInterviewSystem


class RealAdapter:
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, messages):
        # dual_agent passes dict messages; bridge to model invoke format
        prompt = "\n\n".join([str(m.get("content", "")) for m in messages if isinstance(m, dict)])
        return self.llm.invoke([HumanMessage(content=prompt)])


def _real_adapter():
    if not (os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Real model key is required for integration tests.")
    _set_llm(None)
    llm = _get_llm()
    return RealAdapter(llm)


def test_dual_agent_single_turn_real_model():
    adapter = _real_adapter()
    system = DualAgentInterviewSystem(adapter, adapter)
    out = system.process_turn("我做过 Java 开发")
    assert isinstance(out.directive, str) and out.directive
    assert isinstance(out.next_question, str) and out.next_question
    assert isinstance(out.phase, str) and out.phase


def test_dual_agent_logs_real_model():
    adapter = _real_adapter()
    system = DualAgentInterviewSystem(adapter, adapter)
    system.process_turn("我擅长 Python 和 PyTorch")
    assert any(line.startswith("[Supervisor]") for line in system.logs)
    assert any(line.startswith("[Interviewer]") for line in system.logs)
