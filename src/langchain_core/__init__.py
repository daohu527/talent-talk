"""Lightweight shim for `langchain_core.messages` used by tests and the app.

This module exists so the repository can run without installing the full
`langchain_core` package. It provides minimal message classes expected by the
tests and by the code in `src/interview` and `src/api`.
"""

__all__ = ["messages"]
