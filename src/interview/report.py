"""Post-processing: structured extraction and Markdown report generation.

When the interview state machine is marked as finished, call
``generate_report`` to produce a structured JSON summary and a
Markdown-formatted evaluation report from the full chat history.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from langchain_core.messages import BaseMessage, SystemMessage

from src.interview.agent import LLMProtocol
from src.interview.state import InterviewState

# ---------------------------------------------------------------------------
# Extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM = """\
你是一个结构化信息提取助手。
请根据以下完整的面试对话记录，输出一个合法的 JSON 对象，包含以下字段：
{
  "candidate_name": "候选人全名",
  "summary": "候选人主要优势与亮点（两三句话）",
  "skills": ["技能1", "技能2", ...],
  "experience_years": "工作年限",
  "biggest_project": "最大项目描述",
  "expected_salary": "期望薪资",
  "communication_score": 评分(1-10的整数),
  "suggested_level": "建议职级（例如 P5、P6）"
}
只输出合法的 JSON，不要包含任何额外文字或代码块标记。"""

# ---------------------------------------------------------------------------
# Markdown template
# ---------------------------------------------------------------------------

_REPORT_TEMPLATE = """\
# 面试评估报告：{candidate_name}

## 核心摘要

{summary}

## 基本信息

| 项目 | 内容 |
|------|------|
| 工作年限 | {experience_years} |
| 技术栈 | {skills} |
| 期望薪资 | {expected_salary} |
| 最大项目 | {biggest_project} |

## 综合评分

- **沟通能力**：{communication_score}/10
- **建议职级**：{suggested_level}

## 详细对话记录

{chat_log}
"""

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_structured_data(
    chat_history: list[BaseMessage],
    llm: LLMProtocol,
) -> dict[str, Any]:
    """Call the LLM to extract a structured JSON summary from the full chat.

    Parameters
    ----------
    chat_history:
        The complete list of messages (SystemMessage, HumanMessage,
        AIMessage) from the interview session.
    llm:
        Any object with an ``invoke(messages) -> AIMessage`` interface.

    Returns
    -------
    dict
        Parsed JSON data from the LLM, or an empty dict on parse failure.
    """
    messages: list[BaseMessage] = [
        SystemMessage(content=_EXTRACTION_SYSTEM),
        *chat_history,
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def generate_report(
    chat_history: list[BaseMessage],
    llm: LLMProtocol,
    state: Optional[InterviewState] = None,
) -> str:
    """Generate a Markdown-formatted interview evaluation report.

    Parameters
    ----------
    chat_history:
        The complete conversation.
    llm:
        LLM used for structured extraction.
    state:
        Optional InterviewState used as a fallback for missing fields.

    Returns
    -------
    str
        The Markdown report as a string.
    """
    data = extract_structured_data(chat_history, llm)

    # Fall back to state values when the LLM extraction is incomplete
    if state is not None:
        data.setdefault("candidate_name", state.name or "未知")
        data.setdefault("experience_years", state.experience_years or "未知")
        data.setdefault("biggest_project", state.biggest_project or "未知")
        data.setdefault("expected_salary", state.expected_salary or "未知")

    skills_raw = data.get("skills", [])
    skills_str = (
        "、".join(skills_raw) if isinstance(skills_raw, list) else str(skills_raw)
    )

    chat_log = _format_chat_log(chat_history)

    return _REPORT_TEMPLATE.format(
        candidate_name=data.get("candidate_name", "未知"),
        summary=data.get("summary", ""),
        experience_years=data.get("experience_years", "未知"),
        skills=skills_str or "未知",
        expected_salary=data.get("expected_salary", "未知"),
        biggest_project=data.get("biggest_project", "未知"),
        communication_score=data.get("communication_score", "N/A"),
        suggested_level=data.get("suggested_level", "N/A"),
        chat_log=chat_log,
    )


def _format_chat_log(history: list[BaseMessage]) -> str:
    """Render the chat history as a Markdown conversation log."""
    lines = []
    for msg in history:
        role = type(msg).__name__.replace("Message", "")
        lines.append(f"**{role}**: {msg.content}")
    return "\n\n".join(lines)
