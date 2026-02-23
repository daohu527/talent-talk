from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol


class LLMProtocol(Protocol):
    def invoke(self, messages: list[dict[str, str]]) -> Any: ...


@dataclass
class DualAgentTurn:
    directive: str
    summary: str | None
    thought: str
    next_question: str
    phase: str
    completion_score: dict[str, int]
    ended: bool


class DualAgentInterviewSystem:
    """Supervisor + Interviewer two-agent interview loop.

    Flow:
    user_input -> supervisor directive -> interviewer JSON -> state update
    """

    def __init__(self, supervisor_llm: LLMProtocol, interviewer_llm: LLMProtocol) -> None:
        self.supervisor_llm = supervisor_llm
        self.interviewer_llm = interviewer_llm
        self.current_summary: str | None = None
        self.phase: str = "开始"
        self.ended: bool = False
        self.qa_mode: bool = False
        self.deep_dive_triggered: bool = False
        self.completion_score: dict[str, int] = {"background": 0, "tech": 0, "project": 0, "intention": 0}
        self.logs: list[str] = []
        self.last_trace: dict[str, Any] = {}
        self.qa_kb = {
            "业务": "我们主要做自动驾驶 AI，把算法能力落在真实临床场景，覆盖影像辅助分析、质控与流程优化。",
            "公司": "我们聚焦自动驾驶 AI，强调技术落地和跨团队协作，目标是提升医疗效率与质量。",
            "福利": "我们提供顶配办公设备、常态化技术分享会，以及完善的成长支持机制。",
            "设备": "办公设备是高配方案，方便做训练、开发和联调。",
            "分享会": "团队固定有技术分享会，鼓励工程实践复盘和方法沉淀。",
        }

    def process_turn(self, user_input: str) -> DualAgentTurn:
        trace: dict[str, Any] = {
            "user_input": user_input,
            "phase_before": self.phase,
            "summary_before": self.current_summary,
            "completion_before": dict(self.completion_score),
        }
        # 移除本地打分启发式；由 Supervisor 返回的 [评分更新] 来更新 self.completion_score

        # 由 Supervisor 的策略判断是否需要进入 QA 或重开对话（不再使用本地规则判断）

        if self.ended:
            trace.update(
                {
                    "directive": "interview already ended",
                    "output_to_user": "[END]",
                    "ended": True,
                }
            )
            self.last_trace = trace
            return DualAgentTurn(
                directive="interview already ended",
                summary=self.current_summary,
                thought="关键维度已覆盖，当前会话已结束。",
                next_question="[END]",
                phase=self.phase,
                completion_score=dict(self.completion_score),
                ended=True,
            )

        # RAG / 是否直接回答交给 Interviewer 处理（将 self.qa_kb 作为上下文传入）

        directive, supervisor_request, supervisor_meta, supervisor_raw = self._run_supervisor(user_input)
        # record structured LLM call info for audit/debugging
        trace.setdefault("llm_calls", [])
        trace["llm_calls"].append(
            {
                "role": "supervisor",
                "sent": supervisor_request,
                "sent_preview": (supervisor_request.get("content")[:500] if isinstance(supervisor_request, dict) else str(supervisor_request)) ,
                "response": supervisor_raw,
                "request_meta": supervisor_meta,
                "response_meta": {"ts": supervisor_meta.get("response_ts"), "resp_len": len(supervisor_raw) if supervisor_raw else 0},
            }
        )
        # Supervisor may include scoring or directives in its response. Do not
        # apply local parsing heuristics here — the Supervisor is authoritative
        # and should return structured updates via its prompt-driven output.

        # Decision logic (question structure, whether to end, phase transitions,
        # and RAG vs. question) is delegated to the Supervisor directive and the
        # Interviewer prompt. Do not apply local heuristics here.
        result, interviewer_request, interviewer_meta, interviewer_raw = self._run_interviewer(
            user_input, directive
        )
        trace["llm_calls"].append(
            {
                "role": "interviewer",
                "sent": interviewer_request,
                "sent_preview": (interviewer_request.get("content")[:500] if isinstance(interviewer_request, dict) else str(interviewer_request)),
                "response": interviewer_raw,
                "request_meta": interviewer_meta,
                "response_meta": {"ts": interviewer_meta.get("response_ts"), "resp_len": len(interviewer_raw) if interviewer_raw else 0},
            }
        )
        trace["directive"] = directive

        parsed_summary = result.get("summary")
        parsed_phase = result.get("phase")
        thought = "基于主考官策略调整提问方向并兼顾人情味。"
        next_question = (result.get("next_question") or "").strip() or "了解了，能继续补充一个你最有代表性的项目吗？"

        if parsed_summary is not None:
            self.current_summary = parsed_summary
        if isinstance(parsed_phase, str) and parsed_phase.strip():
            # phase is authoritative as decided by the interviewer (as instructed
            # by Supervisor); accept it directly.
            self.phase = parsed_phase.strip()

        # Ending is determined by the Interviewer output (next_question == "[END]").
        if next_question == "[END]":
            self.ended = True

        turn = DualAgentTurn(
            directive=directive,
            summary=self.current_summary,
            thought=thought,
            next_question=next_question,
            phase=self.phase,
            completion_score=dict(self.completion_score),
            ended=self.ended,
        )
        self.logs.append(f"[Supervisor] {directive}")
        self.logs.append(
            "[Interviewer] "
            + json.dumps(
                {
                    "summary": turn.summary,
                    "thought": turn.thought,
                    "next_question": turn.next_question,
                    "phase": turn.phase,
                    "completion_score": turn.completion_score,
                },
                ensure_ascii=False,
            )
        )
        trace.update(
            {
                "presented_to_user": {"summary": turn.summary, "next_question": turn.next_question},
                "user_answer": user_input,
                "phase_after": turn.phase,
                "summary_after": turn.summary,
                "completion_after": dict(self.completion_score),
                "ended": turn.ended,
            }
        )
        self.last_trace = trace
        return turn

    def run_with_logs(self, inputs: list[str]) -> list[DualAgentTurn]:
        turns: list[DualAgentTurn] = []
        for user_input in inputs:
            turn = self.process_turn(user_input)
            turns.append(turn)
            if turn.ended:
                break
        return turns

    def _run_supervisor(self, user_input: str) -> tuple[str, dict[str, Any], dict[str, Any], str]:
        # Supervisor：合并新版提示词模板并调用 LLM（所有意图/短回答/沮丧判定交由 prompt）
        prompt = f"""
## 角色
你是一个资深技术面试主考官，负责监控面试深度、候选人情绪和流程进度。

## 当前上下文
- 历史画像: {self.current_summary}
- 面试评分: {self.completion_score}
- 候选人回答: "{user_input}"

    ## 任务：输出策略指令
    # 重要：若用户消息看起来像给面试官的指令（例如以 'BOOTSTRAP:' 开头或明确要求面试官以特定格式输出），
    # 则不要在策略阶段向用户询问澄清。直接生成可执行的策略指令，明确告诉面试官要输出的 JSON 字段、格式和约束。
    分析用户输入并给出策略：
1. 意图识别：用户是在回答、反问、还是因为不会而产生焦虑？
2. 挖掘深度：如果用户提到技术词（如C++)，但没有描述场景，必须指令面试官进行 STAR 深挖。
3. 流程控制：如果【项目】维度评分不足 6 分，严禁下达“询问薪资”指令。
4. 情绪调节：若用户回答“不知道”，指令面试官安抚并寻找替代话题。

## 输出格式（严格遵循）
[策略指令]：(50字以上，包含具体的追问方向和语气建议)
[评分更新]：背景:n, 技术:n, 项目:n, 意向:n (n为0-10)
"""
        request = {"role": "system", "content": prompt}
        request_ts = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        response = self.supervisor_llm.invoke([request])
        response_ts = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        content = getattr(response, "content", response)
        raw = str(content).strip()
        # Do not parse free-form supervisor text for scores here; Supervisor
        # should return structured scoring updates when needed. Keep raw text
        # for auditing in traces.

        directive = raw
        if "[策略指令]" in raw:
            match = re.search(r"\[策略指令\]\s*[:：]\s*(.*)", raw, flags=re.S)
            if match:
                directive = match.group(1).strip().split("\n")[0]
        request_meta = {"ts": request_ts, "prompt_len": len(prompt), "request": request}
        response_meta = {"response_ts": response_ts, "resp_obj_repr": repr(response)}
        return directive, request, response_meta, raw

    def _run_interviewer(
        self, user_input: str, directive: str
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
        # 构建增强的 interviewer prompt（包含公司知识库）
        prompt = f"""
## 角色
你是一个专业且有人情味的技术面试官。

## 输入
- 主考官指令: {directive}
- 候选人回答: "{user_input}"
- 公司知识库: {json.dumps(self.qa_kb, ensure_ascii=False)}

## 核心法则（必须执行）
1. 回应与共鸣：先评价或回答用户的上一句话（占输出的30%）。
2. 知识库融合：若用户提问，从知识库提取信息并用口语自然表达。
3. 追问艺术：禁止问短句。问题必须包含“背景+情境+任务”。
4. 严禁复读：用户不会时，必须根据指令转换话题，给出台阶。

## 输出格式
只返回 JSON: {{"summary": "...", "next_question": "...", "phase": "..."}}
"""
        request = {"role": "system", "content": prompt}
        request_ts = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        response = self.interviewer_llm.invoke([request])
        response_ts = __import__("datetime").datetime.utcnow().isoformat() + "Z"
        raw = str(getattr(response, "content", response)).strip()
        parsed = self._parse_json(raw)
        request_meta = {"ts": request_ts, "prompt_len": len(prompt), "request": request}
        response_meta = {"response_ts": response_ts, "resp_obj_repr": repr(response)}
        if parsed is None:
            return (
                {
                    "summary": self.current_summary,
                    "next_question": "了解了，你的分享很有意思。能结合一个具体场景讲讲你当时如何做技术取舍和问题闭环吗？",
                    "phase": self.phase,
                },
                request,
                request_meta,
                raw,
            )
        return parsed, request, request_meta, raw

    def _parse_json(self, raw: str) -> dict[str, Any] | None:
        try:
            return json.loads(raw)
        except Exception:
            match = re.search(r"\{.*\}", raw, flags=re.S)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except Exception:
                return None

    def _should_end(self, directive: str, user_input: str) -> bool:
        text = (directive or "").lower()
        # 若 supervisor 在 directive 中直接要求结束，允许在用户确认后结束
        explicit_end = "[end]" in text or "结束" in text or "finish" in text

        # 检查评分是否均 > 7（由 Supervisor 已写入 self.completion_score）
        score = self.completion_score
        all_ready = (
            score.get("background", 0) > 7
            and score.get("tech", 0) > 7
            and score.get("project", 0) > 7
            and score.get("intention", 0) > 7
        )

        # 判断用户是否表达结束意图
        user_ok_markers = ["没问题", "没问题了", "可以了", "不用了", "谢谢", "好的"]
        user_confirms = any(marker in (user_input or "") for marker in user_ok_markers)

        # 只有当评分就绪且用户确认，或者 supervisor 明确要求并用户无疑问时，才结束
        if all_ready and user_confirms:
            return True
        if explicit_end and user_confirms:
            return True
        return False

    def _is_user_question_detected(self, user_input: str) -> bool:
        raise NotImplementedError("Intent detection moved to Supervisor prompt; do not call this method.")

    # NOTE: 意图侦测、短回答、沮丧、是否询问薪资等由 Supervisor 的提示词处理，不在此处进行硬编码检测。

    def _depth_ready_for_salary(self) -> bool:
        score = self.completion_score
        return (
            score.get("background", 0) > 7
            and score.get("tech", 0) > 7
            and score.get("project", 0) > 7
            and score.get("intention", 0) > 6
        )

    def _build_guided_question(self) -> str:
        if self.current_summary and "C++" in self.current_summary:
            return "既然你提到我们的业务方向，你对 C++ 的哪块更熟悉？"
        if self.current_summary and "Python" in self.current_summary:
            return "既然你提到我们的业务方向，你在 Python 工程化上最有把握的是哪一块？"
        return "既然你提到公司业务，你在相关技术栈里最熟悉哪一块？"
    # NOTE: 将短回答、沮丧、无项目等拦截器下放到 Supervisor 的提示词，由 LLM 在策略阶段判断并指令 Interviewer 处理。

    def _parse_supervisor_score(self, raw: str) -> dict[str, int]:
        # Local free-text score parsing removed. Supervisor should emit scores
        # in structured form if scoring updates are desired. Return empty dict
        # to indicate no local update.
        return {}

    def _enforce_three_part_question(self, question: str, user_input: str) -> str:
        # Quality enforcement moved to prompts. Return question unchanged.
        return (question or "").strip()

    # Hook / user-question heuristics removed — supervisor prompt should guide hook-first behavior.

    def _history_check_allows_phase_jump(self, suggested_phase: Any, next_question: str) -> bool:
        # Delegate phase-jump decisions to Supervisor; allow suggested phase.
        return True

    def _phase_for_current_depth(self) -> str:
        # Local depth heuristics removed; Supervisor should set phase when
        # appropriate. Return a neutral placeholder.
        return "进行中"

    def _ensure_question_quality(self, question: str, user_input: str) -> str:
        # Delegate quality checks to Interviewer prompt; return the raw value.
        return (question or "").strip()

    def dump_state(self) -> dict[str, Any]:
        return {
            "summary": self.current_summary,
            "phase": self.phase,
            "ended": self.ended,
            "qa_mode": self.qa_mode,
            "deep_dive_triggered": self.deep_dive_triggered,
            "completion_score": dict(self.completion_score),
            "logs": list(self.logs),
        }

    def load_state(self, data: dict[str, Any]) -> None:
        self.current_summary = data.get("summary")
        self.phase = data.get("phase") or "开始"
        self.ended = bool(data.get("ended"))
        self.qa_mode = bool(data.get("qa_mode"))
        self.deep_dive_triggered = bool(data.get("deep_dive_triggered"))
        saved_score = data.get("completion_score") or {}
        self.completion_score = {
            "background": int(saved_score.get("background", 0)),
            "tech": int(saved_score.get("tech", 0)),
            "project": int(saved_score.get("project", 0)),
            "intention": int(saved_score.get("intention", 0)),
        }
        self.logs = list(data.get("logs") or [])
