"""Minimal LangGraph-style orchestrator for interview flow (demo).

This module implements a tiny graph runner with three node types:
- AskNode: asks a question (calls LLM)
- ExtractNode: extracts structured fields from the latest human reply (calls LLM)
- DecisionNode: inspects the state and decides next node (no LLM required)

The implementation is intentionally lightweight to avoid external
dependencies and to be easy to test/mock.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src.interview.agent import LLMProtocol, _ANALYSIS_SYSTEM, _INTERVIEW_SYSTEM, FIELD_DESCRIPTIONS
from src.interview.state import InterviewState


class Node:
    def run(self, context: dict, llm: LLMProtocol) -> dict:
        raise NotImplementedError()


class ExtractNode(Node):
    """Run the analysis LLM on the latest human message and update state."""

    def run(self, context: dict, llm: LLMProtocol) -> dict:
        history: List[BaseMessage] = context.get("history", [])
        state: InterviewState = context.get("state")
        messages = [SystemMessage(content=_ANALYSIS_SYSTEM), *history]
        resp = llm.invoke(messages)
        raw = resp.content.strip()
        # reuse agent's extraction logic by importing functionally similar code
        try:
            import json, re

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            extracted = json.loads(raw) if raw else {}
        except Exception:
            extracted = {}

        # If extraction failed to produce structured data, try a simple
        # heuristic: if the last human reply is short and the immediate
        # preceding AI message asked for the candidate's name, treat the
        # human reply as the name. This helps when the mock LLM or a
        # noisy extractor doesn't emit JSON but the user simply replied.
        if not extracted:
            try:
                # find last human message
                last_human = None
                last_ai = None
                for msg in reversed(history):
                    if isinstance(msg, HumanMessage) and last_human is None:
                        last_human = msg
                    elif isinstance(msg, AIMessage) and last_ai is None:
                        last_ai = msg
                    if last_human and last_ai:
                        break

                if last_human and last_ai:
                    txt = (getattr(last_ai, "content", "") or "").lower()
                    candidate = (getattr(last_human, "content", "") or "").strip()
                    # heuristic rules: short reply, no question mark, ai asked for name
                    if 0 < len(candidate) <= 30 and "?" not in candidate and ("叫什么名字" in txt or "姓名" in txt or "your name" in txt):
                        extracted = {"name": candidate}
            except Exception:
                pass

        updates = {k: v for k, v in extracted.items() if k in FIELD_DESCRIPTIONS and getattr(state, k, None) is None and v}
        if updates:
            state = state.model_copy(update=updates)

        return {"history": history, "state": state}


class DecisionNode(Node):
    """Decide whether to ask next question or finish."""

    def run(self, context: dict, llm: LLMProtocol) -> dict:
        state: InterviewState = context.get("state")

        # If already complete, short-circuit
        if state.is_complete():
            state = state.model_copy(update={"finished": True})
            return {"action": "end", "state": state}

        # Ask the LLM to decide the next action. Provide the interview
        # checklist and current state, and request a JSON response with
        # fields: action (ask|end), next_field (or null), question (string).
        checklist = "\n".join(f"- {desc}" for desc in FIELD_DESCRIPTIONS.values())
        system_content = _INTERVIEW_SYSTEM.format(checklist=checklist, state=state.to_prompt_str())
        decision_system = (
            system_content
            + "\n\n现在基于上面的任务和当前已收集信息，决定下一步动作。"
            + "\n仅输出一个 JSON 对象，形如: {\"action\":\"ask\"|\"end\", \"next_field\": <字段名或null>, \"question\": <要问的问题或空字符串>}。"
        )

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=decision_system),
            HumanMessage(content="请返回 JSON，不要包含其他说明。"),
        ]

        try:
            resp = llm.invoke(messages)
            raw = (getattr(resp, "content", "") or "").strip()
            import re, json

            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            parsed = json.loads(raw) if raw else {}
            action = parsed.get("action")
            next_field = parsed.get("next_field")
            question = parsed.get("question")
            if action == "end":
                state = state.model_copy(update={"finished": True})
                return {"action": "end", "state": state, "question": question}
            # Ensure next_field is one of known fields
            if next_field not in FIELD_DESCRIPTIONS:
                # fallback to first missing
                missing = state.missing_fields()
                next_field = missing[0] if missing else None
            return {"action": "ask", "next_field": next_field, "question": question, "state": state}
        except Exception:
            # Fallback deterministic behaviour
            missing = state.missing_fields()
            next_field = missing[0] if missing else None
            return {"action": "ask", "next_field": next_field, "state": state}


class AskNode(Node):
    """Generate a single focused question for the next missing field."""

    def __init__(self, template: Optional[str] = None):
        self.template = template

    def run(self, context: dict, llm: LLMProtocol) -> dict:
        state: InterviewState = context.get("state")
        next_field: str = context.get("next_field")
        # If a question was pre-computed by DecisionNode, use it.
        pre_question: str = context.get("question")
        checklist = "\n".join(f"- {desc}" for desc in FIELD_DESCRIPTIONS.values())
        system_content = _INTERVIEW_SYSTEM.format(checklist=checklist, state=state.to_prompt_str())
        if pre_question:
            question = pre_question
        else:
            # craft a short prompt asking for the specific field
            question = f"请提供你的{FIELD_DESCRIPTIONS.get(next_field, next_field)}。一次只回答这一项。"

        messages = [SystemMessage(content=system_content), HumanMessage(content=question)]
        try:
            resp = llm.invoke(messages)
            reply = resp.content
        except Exception:
            reply = question

        return {"reply": reply, "question_for": next_field, "state": state}


class GraphRunner:
    """Run a simple linear graph: Extract -> Decision -> Ask (repeat)"""

    def __init__(self, llm: LLMProtocol):
        self.llm = llm

    def run_once(self, history: List[BaseMessage], state: InterviewState) -> dict:
        ctx = {"history": history, "state": state}
        # Extract
        ctx = ExtractNode().run(ctx, self.llm)
        # Decide
        decision = DecisionNode().run(ctx, self.llm)
        if decision.get("action") == "end":
            return {
                "reply": "谢谢，面试结束。[END]",
                "state": decision.get("state"),
                "last_action": "end",
                "next_field": None,
                "question_for": None,
            }
        # Ask
        ask_ctx = {"state": decision.get("state"), "next_field": decision.get("next_field")}
        ask_result = AskNode().run(ask_ctx, self.llm)
        return {
            "reply": ask_result.get("reply"),
            "state": ask_result.get("state"),
            "last_action": "ask",
            "next_field": ask_result.get("question_for"),
            "question_for": ask_result.get("question_for"),
        }
