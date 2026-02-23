"""Simple session storage for interview conversations.

Provides an in-memory store with an optional Redis backend (configured via
`REDIS_URL`). The store keeps `history` (list of message dicts) and `state`
(`InterviewState.model_dump()` JSON serializable dict).
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

try:
    import redis
except Exception:
    redis = None


class SessionStore:
    def __init__(self) -> None:
        self._mem: dict[str, dict[str, Any]] = {}
        self._redis = None
        url = os.getenv("REDIS_URL")
        if url and redis is not None:
            self._redis = redis.from_url(url)

    def create(self, initial_state: Optional[dict] = None) -> str:
        sid = str(uuid.uuid4())
        payload = {"history": [], "state": initial_state or {}}
        if self._redis:
            self._redis.set(sid, json.dumps(payload))
        else:
            self._mem[sid] = payload
        return sid

    def get(self, sid: str) -> dict[str, Any] | None:
        if self._redis:
            raw = self._redis.get(sid)
            if raw is None:
                return None
            return json.loads(raw)
        return self._mem.get(sid)

    def update(self, sid: str, payload: dict[str, Any]) -> None:
        if self._redis:
            self._redis.set(sid, json.dumps(payload))
        else:
            self._mem[sid] = payload


_store = SessionStore()


def get_store() -> SessionStore:
    return _store
