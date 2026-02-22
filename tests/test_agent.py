"""Tests for the interview agent logic.

The LLM is mocked so these tests run entirely offline without an API key.
"""

from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage

from src.interview.agent import _extract_and_update, process_turn
from src.interview.state import InterviewState


# ---------------------------------------------------------------------------
# Minimal mock LLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Returns a predefined response on every invoke call."""

    def __init__(self, content: str):
        self.content = content
        self.calls: list = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.content)


# ---------------------------------------------------------------------------
# _extract_and_update
# ---------------------------------------------------------------------------


def test_extract_updates_name():
    state = InterviewState()
    llm = MockLLM(json.dumps({"name": "张三"}))
    history = [HumanMessage(content="我叫张三")]
    updated = _extract_and_update(history, state, llm)
    assert updated.name == "张三"


def test_extract_skips_already_known_fields():
    state = InterviewState(name="原名")
    llm = MockLLM(json.dumps({"name": "新名字"}))
    history = [HumanMessage(content="...")]
    updated = _extract_and_update(history, state, llm)
    # Existing value should NOT be overwritten
    assert updated.name == "原名"


def test_extract_handles_invalid_json():
    state = InterviewState()
    llm = MockLLM("这不是 JSON")
    history = [HumanMessage(content="hello")]
    updated = _extract_and_update(history, state, llm)
    assert updated == state  # no change


def test_extract_handles_empty_response():
    state = InterviewState()
    llm = MockLLM("")
    history = [HumanMessage(content="hello")]
    updated = _extract_and_update(history, state, llm)
    assert updated == state


def test_extract_strips_code_fences():
    state = InterviewState()
    llm = MockLLM('```json\n{"experience_years": "3年"}\n```')
    history = [HumanMessage(content="我做了3年开发")]
    updated = _extract_and_update(history, state, llm)
    assert updated.experience_years == "3年"


def test_extract_multiple_fields_at_once():
    state = InterviewState()
    payload = {"name": "李四", "tech_stack": "Go, Kubernetes"}
    llm = MockLLM(json.dumps(payload))
    history = [HumanMessage(content="我叫李四，用 Go 和 K8s")]
    updated = _extract_and_update(history, state, llm)
    assert updated.name == "李四"
    assert updated.tech_stack == "Go, Kubernetes"


# ---------------------------------------------------------------------------
# process_turn
# ---------------------------------------------------------------------------


def _make_full_state(**kwargs) -> InterviewState:
    defaults = dict(
        name="张三",
        experience_years="3年",
        tech_stack="Python",
        biggest_project="电商平台",
        expected_salary="20k",
    )
    defaults.update(kwargs)
    return InterviewState(**defaults)


class DualMockLLM:
    """Returns different responses for successive invocations."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    def invoke(self, messages):
        content = self._responses[self._idx % len(self._responses)]
        self._idx += 1
        return AIMessage(content=content)


def test_process_turn_returns_reply_and_state():
    llm = DualMockLLM(["{}", "请问您叫什么名字？"])
    state = InterviewState()
    history = [HumanMessage(content="你好")]
    reply, new_state = process_turn(history, state, llm)
    assert isinstance(reply, str)
    assert isinstance(new_state, InterviewState)


def test_process_turn_marks_finished_when_complete():
    # extraction returns nothing new; state is already complete
    llm = DualMockLLM(["{}", "感谢您参加面试！[END]"])
    state = _make_full_state()
    history = [HumanMessage(content="期望薪资 20k")]
    reply, new_state = process_turn(history, state, llm)
    assert new_state.finished is True


def test_process_turn_sets_finished_on_end_marker():
    llm = DualMockLLM(["{}", "面试结束，感谢！[END]"])
    state = InterviewState()
    history = [HumanMessage(content="好的")]
    reply, new_state = process_turn(history, state, llm)
    assert new_state.finished is True


def test_process_turn_does_not_overwrite_existing_data():
    llm = DualMockLLM([json.dumps({"name": "新名字"}), "继续提问"])
    state = InterviewState(name="张三")
    history = [HumanMessage(content="我叫张三")]
    _, new_state = process_turn(history, state, llm)
    assert new_state.name == "张三"
