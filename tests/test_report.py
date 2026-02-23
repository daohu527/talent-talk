from __future__ import annotations

import os

from langchain_core.messages import HumanMessage

from src.api.main import _get_llm, _set_llm
from src.interview.report import extract_structured_data, generate_report
from src.interview.state import InterviewState


def _real_llm():
    if not (os.getenv("ARK_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise RuntimeError("Real model key is required for integration tests.")
    _set_llm(None)
    return _get_llm()


def test_extract_structured_data_real_model():
    llm = _real_llm()
    history = [HumanMessage(content="我叫张三，3年经验，技术栈Python和FastAPI")]
    result = extract_structured_data(history, llm)
    assert isinstance(result, dict)


def test_generate_report_real_model():
    llm = _real_llm()
    history = [HumanMessage(content="我叫张三，做Python开发3年")]
    state = InterviewState(
        name="张三",
        experience_years="3年",
        tech_stack="Python",
        biggest_project="电商平台",
        expected_salary="20k",
        finished=True,
    )
    report = generate_report(history, llm, state=state)
    assert isinstance(report, str) and report
    assert "#" in report
