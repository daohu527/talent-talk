import json
import re
from typing import Optional, Callable, Dict, Any


class InterviewSession:
    """Simple JSON-state interview session driven by an LLM callable.

    The LLM callable should accept a single string prompt and return the LLM's
    raw string output (ideally the JSON object required by the system prompt).
    """

    def __init__(self, llm: Callable[[str], str]):
        self.current_summary: Optional[str] = None
        self.llm = llm
        self.system_prompt = (
            "你是一个专业的面试助理。你的任务是通过追问获取求职者的关键信息。\n"
            "1. 必须获取的维度：基本背景、核心技术栈、代表性项目、期望职位。\n"
            "2. 状态维护：每一轮都要在 summary 中更新已掌握的所有信息。\n"
            "3. 约束：每次只问一个问题。严禁输出 JSON 以外的任何字符。\n"
            "4. 结束判定：当上述维度收集完毕，next_question 必须返回 \"[END]\"。\n"
            "输出格式：{\"summary\": \"string\", \"next_question\": \"string\"}"
        )

    def process_turn(self, user_input: str) -> Dict[str, Any]:
        """Process one turn: send prompt to llm, parse JSON, update summary.

        Returns a dict with keys: `summary` (updated), `next_question` (string),
        and on parse failure `raw` (the raw llm output) plus a repair prompt.
        """
        prompt_parts = [self.system_prompt]
        prompt_parts.append(f"Current summary: {self.current_summary}")
        prompt_parts.append(f"User input: {user_input}")
        prompt_parts.append("请严格只返回 JSON，例如 {\"summary\": \"...\", \"next_question\": \"...\"}。")
        prompt = "\n\n".join(prompt_parts)

        raw = self.llm(prompt)
        parsed = self._parse_json(raw)
        if parsed is None:
            # Graceful fallback: ask model (caller) to return strict JSON.
            return {
                "summary": self.current_summary,
                "next_question": "请仅返回 JSON，格式 {\"summary\":..., \"next_question\":...}。",
                "raw": raw,
            }

        # Accept explicit null -> None
        summary = parsed.get("summary")
        next_q = parsed.get("next_question")

        if summary is not None:
            self.current_summary = summary

        return {"summary": self.current_summary, "next_question": next_q}

    def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        """Try strict json.loads, then fallback to extracting first {...} block."""
        try:
            return json.loads(raw)
        except Exception:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
