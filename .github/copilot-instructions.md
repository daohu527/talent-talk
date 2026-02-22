# Project Instructions for AI Assistants

Purpose: give concise, repository-level guidance so agents produce reviewable, testable work.

Basic rules

- Follow repository linters and formatters before opening PRs (run commands listed below).
- Prefer small, focused changes with one responsibility per issue/PR.
- Always add or update tests for behavioral changes; include commands to run them.
- Limit edits to paths listed in the issue `Scope & Constraints`.
- Preserve backwards compatibility unless the issue explicitly requests a breaking change.

Agent workflow specifics

- Link the originating `agent` issue in the PR body and reference acceptance criteria.
- If task requirements are ambiguous, ask up to 3 clarifying questions before modifying code.
- Keep diffs minimal and reversible; prefer multiple small PRs over a single large one.
- When changing APIs, include migration notes and update docs/changelog.

Local verification (examples)

```bash
# JS/Node projects
npm ci && npm test && npm run lint

# Python projects
python -m pip install -r requirements-dev.txt
pytest
flake8
```

If the repository uses other tooling, include equivalent commands in the issue body.

If you are an agent: prefer explicit, test-driven changes and surface any uncertainties as questions in the issue or PR.
