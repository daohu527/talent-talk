# AI Collaboration Template

This repository contains a lightweight template for assigning, tracking, and iterating work intended for AI assistants (e.g., Copilot coding agents).

Contents:
- `.github/ISSUE_TEMPLATE/agent-task.md` — primary issue template for agent tasks
- `.github/copilot-instructions.md` — repository-level guidance for AI assistants
- `.github/ISSUE_TEMPLATE/plan-issue.md` — a plan-style issue template for complex work
- `.github/PULL_REQUEST_TEMPLATE/agent-pr.md` — PR checklist for agent-generated PRs

How to use:

1. Create small, testable issues using the `agent-task` template.
2. Attach acceptance criteria and allowed file paths to limit scope.
3. Require tests and CI checks to validate agent output.

Iterating the templates:

- After each agent-driven PR, update the templates with examples and clarifications that improved results.
- Add project-specific rules to `.github/copilot-instructions.md` so AI follows team conventions.

This template is minimal — adapt and expand it to match your stack and workflows.
