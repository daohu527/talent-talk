"""Minimal message classes for top-level shim package."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaseMessage:
    content: str


class HumanMessage(BaseMessage):
    pass


class AIMessage(BaseMessage):
    pass


class SystemMessage(BaseMessage):
    pass

__all__ = ["BaseMessage", "HumanMessage", "AIMessage", "SystemMessage"]
