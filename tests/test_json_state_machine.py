import os

from langchain_core.messages import HumanMessage

from src.api.main import _get_llm, _set_llm
from src.interview.session import InterviewSession


def _real_session():
    if not (os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Real model key is required for integration tests.")
    _set_llm(None)
    llm = _get_llm()

    def call(prompt: str) -> str:
        resp = llm.invoke([HumanMessage(content=prompt)])
        return getattr(resp, "content", str(resp))

    return InterviewSession(llm=call)


def test_json_state_machine_with_real_model_runs():
    sess = _real_session()
    out1 = sess.process_turn("")
    assert "next_question" in out1

    out2 = sess.process_turn("我是王泽，做算法的。")
    assert "next_question" in out2

    out3 = sess.process_turn("我擅长 Python 和 PyTorch。")
    assert "next_question" in out3
