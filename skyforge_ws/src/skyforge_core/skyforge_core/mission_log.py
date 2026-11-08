"""mission_logs: every decision with its inputs, outputs and a one-line sentence.

Events are plain dicts: {"ts", "type", "phase", "msg", ...details}. The
`msg` field is the templated sentence shown in the live decision feed.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional


class MissionLog:
    def __init__(self, path: Optional[str] = None, keep: int = 5000):
        self.path = path
        self.events: list[dict[str, Any]] = []
        self.keep = keep
        self._fh = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._fh = open(path, "w", encoding="utf-8")

    def add(self, ev: dict[str, Any]) -> None:
        self.events.append(ev)
        if len(self.events) > self.keep:
            self.events = self.events[-self.keep :]
        if self._fh:
            self._fh.write(json.dumps(ev, default=str, ensure_ascii=False) + "\n")
            self._fh.flush()

    def extend(self, evs) -> None:
        for e in evs:
            self.add(e)

    def of_type(self, t: str) -> list[dict]:
        return [e for e in self.events if e.get("type") == t]

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None


def event(ts: float, etype: str, phase: str, msg: str, **details) -> dict:
    return {"ts": round(ts, 3), "type": etype, "phase": phase, "msg": msg, **details}
