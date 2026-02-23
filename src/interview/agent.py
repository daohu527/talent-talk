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
你是一位资深的 HR 面试官（专业、友好、善于引导）。
你的任务是通过对话逐步收集候选人的下列信息：
{checklist}

行为要求：
1) 每次只询问一个信息项。一次只问一件事。
2) 如果候选人回答过于简短或只给出关键词（例如对「最大项目」只说「电商系统」），请识别为不充分回答并追问具体细节：比如“能具体讲讲在这个电商系统中，你负责的核心模块和遇到的最大技术难点吗？”。
3) 如果候选人确实答不上来，应先安抚（例如“没关系，这个问题比较偏，我们聊聊下一个……”），而不是机械地跳过或直接切换话题。
4) 在收集完所有清单项后，礼貌结束面试并输出标记 [END]，随后询问“你有什么想问我们的？”以进入候选人反向提问环节。

当前已收集信息：
{state}
"""


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
    # --- New flow: LLM-driven Chain-of-Thought step -------------------------
    # Ask the LLM to (internally) perform analysis, decide next action,
    # and produce a next question. The LLM must return a JSON object with
    # keys: extracted (dict), action ("ask"|"end"), next_field (or null),
    # question (string).
    cot = _cot_step(history, current_state, llm)

    # Apply extracted updates (only fill previously empty fields)
    extracted = cot.get("extracted") or {}
    updates = {
        k: v
        for k, v in extracted.items()
        if k in FIELD_DESCRIPTIONS and getattr(current_state, k, None) is None and v
    }
    if updates:
        current_state = current_state.model_copy(update=updates)

    # Apply action
    action = cot.get("action") or "ask"
    question = cot.get("question") or ""
    # If action says end, mark finished and return closing text
    if action == "end":
        if not current_state.finished:
            current_state = current_state.model_copy(update={"finished": True})
        reply = question or "谢谢，面试结束。[END]"
        return reply, current_state

    # Otherwise, return the question asked by the LLM (or fallback)
    if not question:
        # fallback to previous deterministic generation
        reply = _generate_reply(history, current_state, llm)
    else:
        reply = question

    # If the LLM signaled [END] in text, mark finished
    if "[END]" in (reply or "") and not current_state.finished:
        current_state = current_state.model_copy(update={"finished": True})

    return reply, current_state


def _cot_step(
    history: Sequence[BaseMessage],
    state: InterviewState,
    llm: LLMProtocol,
) -> dict:
    """Run a single LLM-driven Chain-of-Thought step.

    The LLM is asked (internally) to: extract any newly mentioned fields,
    decide whether to ask another question or end the interview, and
    produce the next question text. The LLM must return only a JSON
    object. Do not reveal internal reasoning.
    """
    checklist = "\n".join(f"- {desc}" for desc in FIELD_DESCRIPTIONS.values())
    system_content = _INTERVIEW_SYSTEM.format(checklist=checklist, state=state.to_prompt_str())

    instruction = (
        system_content
           + "\n\n现在请一次性完成下列任务并只输出一个合法的 JSON 对象：\n"
           + "1) extracted: 从用户最近的回答中提取本轮出现的新字段（json 对象），\n"
           + "2) action: \"ask\" 或 \"end\"，表示下一步动作，\n"
           + "3) next_field: 如果 action==\"ask\"，请填写要收集的字段名（例如 \"name\"），否则为 null，\n"
           + "4) question: 如果 action==\"ask\"，请给出面试官要问的礼貌一句话问题（一次只问一项），如果 action==\"end\"，可给出结束语或留空。\n\n"
           + "示例输出：\n```\n"
           + "{\n"
           + "  \"extracted\": {\"name\": \"张三\", \"tech_stack\": \"Python, React\"},\n"
           + "  \"action\": \"ask\",\n"
           + "  \"next_field\": \"experience_years\",\n"
           + "  \"question\": \"请问您的工作年限是多少？\"\n"
           + "}\n"
           + "```\n仅输出 JSON，不要包含任何其他解释或推理文本。内部可以进行链式思考，但不要泄露推理过程。若无法确定字段，请把该字段留空或不包含在 extracted 中。"
    )

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [SystemMessage(content=instruction), *history, HumanMessage(content="请只返回 JSON。")]

    try:
        resp = llm.invoke(messages)
        raw = (getattr(resp, "content", "") or "").strip()
        import re, json

        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        parsed = json.loads(raw) if raw else {}
        # Normalize keys
        return {
            "extracted": parsed.get("extracted") or parsed.get("extraction") or {},
            "action": parsed.get("action"),
            "next_field": parsed.get("next_field"),
            "question": parsed.get("question"),
        }
    except Exception:
        # fallback: no extraction, ask for first missing field
        missing = state.missing_fields()
        nf = missing[0] if missing else None
        return {"extracted": {}, "action": "ask", "next_field": nf, "question": ""}


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
