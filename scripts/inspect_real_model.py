from src.api.main import _get_llm, _DualAgentLLMAdapter
from src.interview.dual_agent import DualAgentInterviewSystem
import json

llm = _get_llm()
adapter = _DualAgentLLMAdapter(llm)
system = DualAgentInterviewSystem(adapter, adapter)

# Single turn: user says they are a student studying C++
user_input = "我是一名大学生，我在学习 c++"
turn = system.process_turn(user_input)

print(json.dumps({
    "next_question": turn.next_question,
    "summary": turn.summary,
    "phase": turn.phase,
    "ended": turn.ended,
}, ensure_ascii=False, indent=2))

# Print last_trace llm_calls responses (short)
trace = system.last_trace
llm_calls = trace.get("llm_calls", [])
for call in llm_calls:
    print("--- LLM CALL ---")
    print("role:", call.get("role"))
    print("sent_preview:", call.get("sent_preview"))
    print("response:", call.get("response"))
    print()
