#!/usr/bin/env python3
from src.api.main import demo_api, DemoRequest, demo_session_evaluation

# Simple client to run a short demo sequence and print session id

# bootstrap (creates session)
resp = demo_api(DemoRequest(text=""))
print("session_id:", resp.session_id)
# continue a couple of turns
texts = [
    "我在开始学习C++",
    "我有一些项目经验，主要是影像处理",
    "谢谢，面试结束。[END]",
]
for t in texts:
    resp = demo_api(DemoRequest(text=t, session_id=resp.session_id))
    print("round reply:", resp.next_question)

# print evaluation / logs_file
eval_res = demo_session_evaluation(resp.session_id)
print("log file:", eval_res.get("logs_file"))
