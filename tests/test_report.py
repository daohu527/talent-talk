"""Tests for the report generation module."""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from src.interview.report import (
    _format_chat_log,
    extract_structured_data,
    generate_report,
)
from src.interview.state import InterviewState


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, messages):
        return AIMessage(content=self.content)


# ---------------------------------------------------------------------------
# extract_structured_data
# ---------------------------------------------------------------------------

_SAMPLE_DATA = {
    "candidate_name": "张三",
    "summary": "候选人具备扎实的 Python 基础。",
    "skills": ["Python", "Django"],
    "experience_years": "3年",
    "biggest_project": "电商平台",
    "expected_salary": "20k",
    "communication_score": 8,
    "suggested_level": "P6",
}


def test_extract_structured_data_parses_json():
    llm = MockLLM(json.dumps(_SAMPLE_DATA))
    history = [HumanMessage(content="我叫张三")]
    result = extract_structured_data(history, llm)
    assert result["candidate_name"] == "张三"
    assert result["skills"] == ["Python", "Django"]
    assert result["communication_score"] == 8


def test_extract_structured_data_strips_code_fences():
    llm = MockLLM(f'```json\n{json.dumps(_SAMPLE_DATA)}\n```')
    history = []
    result = extract_structured_data(history, llm)
    assert result["candidate_name"] == "张三"


def test_extract_structured_data_returns_empty_on_invalid_json():
    llm = MockLLM("不是 JSON")
    result = extract_structured_data([], llm)
    assert result == {}


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


def test_generate_report_contains_candidate_name():
    llm = MockLLM(json.dumps(_SAMPLE_DATA))
    history = [HumanMessage(content="我叫张三")]
    report = generate_report(history, llm)
    assert "张三" in report


def test_generate_report_contains_skills():
    llm = MockLLM(json.dumps(_SAMPLE_DATA))
    report = generate_report([], llm)
    assert "Python" in report
    assert "Django" in report


def test_generate_report_contains_score():
    llm = MockLLM(json.dumps(_SAMPLE_DATA))
    report = generate_report([], llm)
    assert "8" in report


def test_generate_report_falls_back_to_state():
    llm = MockLLM("{}")  # LLM returns no data
    state = InterviewState(
        name="李四",
        experience_years="5年",
        biggest_project="支付系统",
        expected_salary="30k",
    )
    report = generate_report([], llm, state=state)
    assert "李四" in report
    assert "5年" in report


def test_generate_report_is_markdown():
    llm = MockLLM(json.dumps(_SAMPLE_DATA))
    report = generate_report([], llm)
    assert report.startswith("#")
    assert "##" in report


# ---------------------------------------------------------------------------
# _format_chat_log
# ---------------------------------------------------------------------------


def test_format_chat_log_includes_roles():
    history = [
        HumanMessage(content="你好"),
        AIMessage(content="你好，请问您叫什么名字？"),
    ]
    log = _format_chat_log(history)
    assert "Human" in log
    assert "AI" in log
    assert "你好" in log
