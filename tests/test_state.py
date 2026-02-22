"""Tests for InterviewState model."""

from src.interview.state import InterviewState


def test_default_state_all_none():
    state = InterviewState()
    assert state.name is None
    assert state.experience_years is None
    assert state.tech_stack is None
    assert state.biggest_project is None
    assert state.expected_salary is None
    assert state.finished is False


def test_missing_fields_returns_all_when_empty():
    state = InterviewState()
    missing = state.missing_fields()
    assert set(missing) == {"name", "experience_years", "tech_stack",
                            "biggest_project", "expected_salary"}


def test_missing_fields_excludes_populated():
    state = InterviewState(name="张三", tech_stack="Python")
    missing = state.missing_fields()
    assert "name" not in missing
    assert "tech_stack" not in missing
    assert "experience_years" in missing


def test_is_complete_false_when_partial():
    state = InterviewState(name="张三")
    assert state.is_complete() is False


def test_is_complete_true_when_all_filled():
    state = InterviewState(
        name="张三",
        experience_years="3年",
        tech_stack="Python",
        biggest_project="电商系统",
        expected_salary="20k",
    )
    assert state.is_complete() is True


def test_finished_field_not_in_missing_fields():
    state = InterviewState()
    assert "finished" not in state.missing_fields()


def test_to_prompt_str_contains_all_fields():
    state = InterviewState(name="张三")
    prompt = state.to_prompt_str()
    assert "张三" in prompt
    assert "已知" in prompt
    assert "未知" in prompt
    assert "finished" not in prompt


def test_model_copy_update():
    state = InterviewState()
    updated = state.model_copy(update={"name": "李四"})
    assert updated.name == "李四"
    assert state.name is None  # original unchanged
