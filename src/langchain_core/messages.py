"""Minimal message classes used by the test-suite and the app.

Only `content` is required by the existing code; these classes mirror the
small subset of behavior the repo expects from LangChain-style messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class BaseMessage:
    content: str


class HumanMessage(BaseMessage):
    pass


class AIMessage(BaseMessage):
    pass


class SystemMessage(BaseMessage):
    pass


# Keep module exports similar to the upstream package
__all__ = ["BaseMessage", "HumanMessage", "AIMessage", "SystemMessage"]
