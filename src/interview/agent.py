"""Core interview agent implementing the checklist-based Chain-of-Thought loop.

Flow per turn
-------------
Step A – Analyse the candidate's latest reply and extract any new information.
Step B – Update the InterviewState checklist.
Step C – Decide whether the interview is finished or which field is still missing.
Step D – Generate the next polite, single-focused question (or a closing message).

The LLM instance is injected so callers can swap it for a mock in tests.
"""

from __future__ import annotations

import json
import re
from typing import Protocol, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from src.interview.state import InterviewState

# ---------------------------------------------------------------------------
# LLM protocol – allows dependency injection / mocking
# ---------------------------------------------------------------------------

FIELD_DESCRIPTIONS = {
    "name": "候选人姓名",
    "experience_years": "工作年限",
    "tech_stack": "技术栈",
    "biggest_project": "做过的最大项目",
    "expected_salary": "期望薪资",
}


class LLMProtocol(Protocol):
    """Minimal interface required from the language model."""

    def invoke(self, messages: list[BaseMessage]) -> AIMessage: ...


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_ANALYSIS_SYSTEM = """\
你是一个信息提取助手。
给定一段面试对话和候选人刚才的回答，请从中提取以下候选人信息（如果提到的话）：
- name: 候选人姓名
- experience_years: 工作年限（字符串，例如 "3年"）
- tech_stack: 技术栈（字符串，例如 "Python, React"）
- biggest_project: 做过的最大项目（简短描述）
- expected_salary: 期望薪资

只输出一个合法的 JSON 对象，仅包含本次回答中**新出现**的字段，没有则输出 {}.
不要包含任何额外文字或代码块标记。"""

_INTERVIEW_SYSTEM = """\
你是一位经验丰富、态度亲和的技术面试官。
你的任务是通过自然对话逐步收集候选人的以下信息：
{checklist}

规则：
1. 每次只问一个问题，不要一次问多个。
2. 如果候选人回答过于简短或模糊，请追问具体细节。
3. 如果候选人答不上来，给予鼓励后跳到下一个问题。
4. 当所有信息都收集完毕后，礼貌地结束面试，并在最后输出标记 [END]。
5. 面试结束前询问候选人是否有问题想问。

当前已收集信息：
{state}"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def process_turn(
    history: list[BaseMessage],
    current_state: InterviewState,
    llm: LLMProtocol,
) -> tuple[str, InterviewState]:
    """Process one conversation turn and return (ai_reply, updated_state).

    Parameters
    ----------
    history:
        The full conversation history so far, including the latest
        HumanMessage that the candidate just sent.
    current_state:
        The InterviewState populated with whatever has been gathered so far.
    llm:
        Any object with an ``invoke(messages) -> AIMessage`` interface.

    Returns
    -------
    tuple[str, InterviewState]
        The AI's next reply and the updated state after this turn.
    """
    # --- Step A & B: extract new info from the latest human reply ----------
    updated_state = _extract_and_update(history, current_state, llm)

    # --- Step C: check completion ------------------------------------------
    if updated_state.is_complete() and not updated_state.finished:
        updated_state = updated_state.model_copy(update={"finished": True})

    # --- Step D: generate next question or closing message ----------------
    reply = _generate_reply(history, updated_state, llm)

    # If the LLM signals [END] in its reply, mark finished
    if "[END]" in reply and not updated_state.finished:
        updated_state = updated_state.model_copy(update={"finished": True})

    return reply, updated_state


def _extract_and_update(
    history: Sequence[BaseMessage],
    state: InterviewState,
    llm: LLMProtocol,
) -> InterviewState:
    """Run Step A+B: analyse the latest reply and update the state."""
    messages: list[BaseMessage] = [
        SystemMessage(content=_ANALYSIS_SYSTEM),
        *history,
    ]
    response = llm.invoke(messages)
    raw = response.content.strip()

    # Strip markdown code fences if the LLM wraps the JSON
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        extracted: dict = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        extracted = {}

    # Only update fields that are still None to avoid overwriting good data
    updates = {
        k: v
        for k, v in extracted.items()
        if k in FIELD_DESCRIPTIONS and getattr(state, k, None) is None and v
    }
    if updates:
        return state.model_copy(update=updates)
    return state


def _generate_reply(
    history: Sequence[BaseMessage],
    state: InterviewState,
    llm: LLMProtocol,
) -> str:
    """Run Step D: generate the interviewer's next message."""
    checklist = "\n".join(
        f"- {desc}" for desc in FIELD_DESCRIPTIONS.values()
    )
    system_content = _INTERVIEW_SYSTEM.format(
        checklist=checklist,
        state=state.to_prompt_str(),
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=system_content),
        *history,
    ]
    response = llm.invoke(messages)
    return response.content
