TalentTalk — 本地验证与 ARK Demo
=================================

本文档总结了我在仓库中执行的验证步骤、如何本地运行 demo，以及我为解决 `uvicorn` 启动时报错所做的更改。

1) 验证步骤摘要
-----------------
- 我检查了测试文件（`tests/`）以确认期望的接口（例如：需要 `AIMessage`, `HumanMessage`）。
- 为避免在没有安装完整 `langchain_core` 包的情况下启动失败，我添加了一个轻量 shim：`src/langchain_core/messages.py`，提供 `BaseMessage`, `HumanMessage`, `AIMessage`, `SystemMessage`。
- 在 `src/api/main.py` 中新增了 ARK/OpenAI 客户端的懒加载实现与一个可访问的 demo 页面（`/demo`）和 API（`/demo_api`）。

2) 为什么添加 shim
---------------------
运行 `uvicorn src.api.main:app` 时出现错误：

  ModuleNotFoundError: No module named 'langchain_core'

为保证仓库能在没有额外依赖时启动（例如测试环境或开发者机器），我添加了 `src/langchain_core/messages.py`，提供上游包所需的最小接口。

3) 如何本地运行 demo
----------------------
安装依赖（示例）：

```bash
python -m pip install fastapi uvicorn openai
```

设置环境变量并启动服务：

```bash
export ARK_API_KEY='your_ark_api_key_here'
export ARK_BASE_URL='https://ark.cn-beijing.volces.com/api/v3'  # 可选
export ARK_MODEL='doubao-seed-2-0-mini-260215'               # 可选
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

打开浏览器访问 http://localhost:8000/demo，输入图片 URL 与问题文本，页面会调用后端 `/demo_api` 并显示模型返回结果。

4) 修复点清单（已完成）
- 添加 `src/langchain_core/messages.py` shim，解决 ModuleNotFoundError。
- 在 `src/api/main.py` 中添加 ARK client 懒加载和 demo 路由。

5) 后续建议
- 在仓库中加入 `requirements-dev.txt` 并在 CI 中运行测试（我可以为你添加 GitHub Actions 工作流）。
- 根据你实际的 ARK/OpenAI SDK 输出结构完善 `/demo_api` 的返回解析。
- 为 demo 页面增加加载状态、错误提示与基本输入校验。

如果你希望我继续，我可以：
- 添加 CI（GitHub Actions）来自动运行测试；或
- 改进 `/demo_api` 的 response 解析；或
- 增强 demo 页面 UX。
