"""Mission events / state -> compact link message bodies (< 1 KB each)."""
from __future__ import annotations

from typing import Optional


def _r(v, nd=1):
    return [round(float(x), nd) for x in v]


def event_to_msg(ev: dict) -> Optional[tuple[str, dict]]:
    et = ev.get("type")
    msg = str(ev.get("msg", ""))
    if et == "det":
        tr = ev.get("track", {})
        return "det", {
            "id": tr.get("id"), "c": tr.get("cls"), "u": tr.get("category"), "p": _r(tr.get("pos", [0, 0]) + [0]),
            "vd": _r(tr.get("vel", [0, 0]), 2), "cf": round(float(tr.get("conf", 0)), 2), "fl": int(bool(tr.get("floating"))),
        }
    if et == "may":
        return "may", {"f": str(ev.get("fault", ""))[:80], "s": ev.get("snapshot", {})}
    if et == "dec":
        body = {"d": ev.get("decision", "x"), "m": msg[:200]}
        cands = ev.get("candidates")
        if cands:
            body["ch"] = ev.get("chosen")
            body["cand"] = [[c["id"], c["S"], int(c["ok"])] for c in cands[:8]]
        return "dec", body
    if et in ("phase", "goal", "nav", "link", "plan", "security"):
        return "dec", {"d": et, "m": msg[:200]}
    return None
